// hiradix_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using torch::indexing::Slice;

// ---------- Small helpers ----------
static inline double now_monotonic() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static inline void ensure_int64_1d(const torch::Tensor &t) {
  TORCH_CHECK(t.defined(), "Tensor must be defined");
  TORCH_CHECK(t.dtype() == torch::kInt64, "dtype must be int64");
  TORCH_CHECK(t.dim() == 1, "tensor must be 1-D");
}

struct PageKey {
  // First token (page_size==1) or first page tokens (size==page_size)
  std::vector<int64_t> t;
  PageKey() = default;
  PageKey(const int64_t *p, size_t n) : t(p, p + n) {}
  bool operator==(const PageKey &o) const noexcept { return t == o.t; }
};

struct PageKeyHash {
  size_t operator()(const PageKey &k) const noexcept {
    uint64_t h = 1469598103934665603ULL;
    for (auto v : k.t) {
      uint64_t x = static_cast<uint64_t>(v);
      h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return static_cast<size_t>(h);
  }
};

struct TreeNode {
  // Tree topology
  TreeNode *parent = nullptr;
  std::unordered_map<PageKey, TreeNode *, PageKeyHash> children;

  // Key chunk and values
  std::vector<int64_t> key; // token IDs for this edge/path slice
  torch::Tensor value;      // device kv indices (int64); undefined => evicted
  torch::Tensor
      host_value; // host kv indices (int64); undefined => not backed up
  std::vector<std::string>
      hash_value; // per-page cumulative hashes (for storage)

  // Metadata
  int lock_ref = 0;
  int hit_count = 0;
  int host_ref_counter = 0; // protect host_value
  bool backuped_storage = false;
  double last_access_time = now_monotonic();

  int64_t id = 0; // stable node id

  inline bool evicted() const { return !value.defined() || value.numel() == 0; }
  inline bool backuped() const {
    return host_value.defined() && host_value.numel() > 0;
  }

  inline void touch() { last_access_time = now_monotonic(); }

  inline void protect_host() { host_ref_counter++; }
  inline void release_host() {
    if (host_ref_counter <= 0)
      throw std::runtime_error("Host ref already zero.");
    host_ref_counter--;
  }
  inline py::object get_last_hash_value() const {
    if (hash_value.empty())
      return py::none();
    return py::cast(hash_value.back());
  }

  bool is_root() const { return parent == nullptr; }
  bool is_leaf() const { return !is_root() && children.empty(); }
  bool is_leaf_device() const {
    if (is_root() || evicted())
      return false;
    if (children.empty())
      return true;
    return std::all_of(children.begin(), children.end(), [](const auto &kv) {
      const TreeNode *c = kv.second;
      return c && c->evicted();
    });
  }
};

// ---------- Core class ----------
class HiRadixTreeCore {
public:
  explicit HiRadixTreeCore(int page_size, torch::Device device)
      : page_size_(page_size), device_(std::move(device)) {}

  void reset() {
    nodes_.clear();
    next_id_ = 0;
    root_ = make_node();
    root_->lock_ref = 1; // never evict root
    evictable_size_ = 0;
    protected_size_ = 0;
  }

  py::tuple match_prefix(const std::vector<int64_t> &key_in) {
    std::vector<int64_t> key = page_align(key_in);
    auto [vals, last_node] = match_prefix_helper(root_, key);

    int64_t total_matched = 0;
    for (auto &t : vals)
      total_matched += t.numel();
    torch::Tensor out = torch::empty(
        {total_matched},
        torch::TensorOptions().dtype(torch::kInt64).device(device_));
    int64_t offset = 0;
    for (auto &t : vals) {
      if (!t.defined() || t.numel() == 0)
        continue;
      out.narrow(0, offset, t.numel()).copy_(t);
      offset += t.numel();
    }

    TreeNode *last_host_node = last_node;
    int64_t host_hit_length = 0;
    TreeNode *dev_node = last_node;
    while (dev_node && dev_node->evicted()) {
      if (dev_node->backuped())
        host_hit_length += dev_node->host_value.size(0);
      dev_node = dev_node->parent;
    }
    TreeNode *last_device_node = dev_node ? dev_node : root_;

    return py::make_tuple(out, last_device_node, last_host_node,
                          host_hit_length);
  }

  int64_t inc_lock_ref(TreeNode *node) {
    if (!node)
      return 0;
    int64_t delta = 0;
    while (node && node != root_) {
      if (node->lock_ref == 0 && !node->evicted()) {
        evictable_size_ -= node->value.size(0);
        protected_size_ += node->value.size(0);
        delta -= node->value.size(0);
      }
      node->lock_ref += 1;
      node = node->parent;
    }
    return delta;
  }
  int64_t dec_lock_ref(TreeNode *node) {
    if (!node)
      return 0;
    int64_t delta = 0;
    while (node && node != root_) {
      if (node->lock_ref == 1 && !node->evicted()) {
        evictable_size_ += node->value.size(0);
        protected_size_ -= node->value.size(0);
        delta += node->value.size(0);
      }
      node->lock_ref -= 1;
      node = node->parent;
    }
    return delta;
  }

  int64_t evictable_size() const { return evictable_size_; }
  int64_t protected_size() const { return protected_size_; }

  torch::Tensor evict_device(int64_t num_tokens) {
    auto cmp = [](TreeNode *a, TreeNode *b) {
      return a->last_access_time > b->last_access_time; // min-heap by time
    };
    std::priority_queue<TreeNode *, std::vector<TreeNode *>, decltype(cmp)>
        leaves(cmp);

    auto seed = collect_leaves_device();
    for (auto *n : seed)
      leaves.push(n);

    int64_t total_to_evict = 0;
    std::vector<torch::Tensor> indices_to_free;
    indices_to_free.reserve(64);

    while (total_to_evict < num_tokens && !leaves.empty()) {
      TreeNode *x = leaves.top();
      leaves.pop();
      if (x->lock_ref > 0)
        continue;

      indices_to_free.push_back(x->value);
      total_to_evict += x->value.size(0);
      evictable_size_ -= x->value.size(0);
      x->value = torch::Tensor();

      TreeNode *p = x->parent;
      if (!x->backuped()) {
        delete_leaf(x);
      }
      if (p && p->is_leaf_device())
        leaves.push(p);
    }

    if (total_to_evict == 0) {
      return torch::empty(
          {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    torch::Tensor out = torch::empty({total_to_evict}, opts);
    int64_t off = 0;
    for (auto &t : indices_to_free) {
      const auto len = t.size(0);
      out.narrow(0, off, len).copy_(t);
      off += len;
    }
    return out;
  }

  torch::Tensor evict_host(int64_t num_tokens) {
    auto cmp = [](TreeNode *a, TreeNode *b) {
      return a->last_access_time > b->last_access_time; // min-heap by time
    };
    std::priority_queue<TreeNode *, std::vector<TreeNode *>, decltype(cmp)>
        leaves(cmp);

    auto seed = collect_leaves_all();
    for (auto *n : seed)
      leaves.push(n);

    std::vector<torch::Tensor> host_indices_to_free;
    host_indices_to_free.reserve(64);
    int64_t total_to_evict = 0;

    while (total_to_evict < num_tokens && !leaves.empty()) {
      TreeNode *x = leaves.top();
      leaves.pop();

      if (!x)
        continue;
      if (!x->evicted())
        continue; // only host-backed leaves
      if (x->host_ref_counter > 0)
        continue; // protected on host

      host_indices_to_free.push_back(x->host_value);
      total_to_evict += x->host_value.size(0);
      TreeNode *p = x->parent;
      delete_leaf(x);
      if (p && p->is_leaf() && p->evicted()) {
        leaves.push(p);
      }
    }

    if (total_to_evict == 0) {
      return torch::empty(
          {0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    }

    torch::Device host_dev = host_indices_to_free.front().device();
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(host_dev);
    torch::Tensor out = torch::empty({total_to_evict}, opts);
    int64_t off = 0;
    for (auto &t : host_indices_to_free) {
      const auto len = t.size(0);
      out.narrow(0, off, len).copy_(t);
      off += len;
    }
    return out;
  }

  // Split 'child' into new_node -> child at split_len
  TreeNode *split_node(TreeNode *child, int64_t split_len) {
    auto *new_node = make_node();
    // move topology
    new_node->parent = child->parent;
    new_node->lock_ref = child->lock_ref;
    new_node->hit_count = child->hit_count;

    // set new_node key/value/host/hash (prefix part)
    new_node->key.assign(child->key.begin(), child->key.begin() + split_len);

    if (!child->evicted()) {
      new_node->value = child->value.index({Slice(0, split_len)});
      child->value =
          child->value.index({Slice(split_len, torch::indexing::None)});
    }
    if (child->backuped()) {
      new_node->host_value = child->host_value.index({Slice(0, split_len)});
      child->host_value =
          child->host_value.index({Slice(split_len, torch::indexing::None)});
    }
    if (!child->hash_value.empty()) {
      int64_t pages = split_len / page_size_;
      new_node->hash_value.assign(child->hash_value.begin(),
                                  child->hash_value.begin() + pages);
      child->hash_value.erase(child->hash_value.begin(),
                              child->hash_value.begin() + pages);
    }

    PageKey ckey = child_key_from(new_node->key);
    for (auto it = new_node->parent->children.begin();
         it != new_node->parent->children.end(); ++it) {
      if (it->second == child) {
        new_node->parent->children.erase(it);
        break;
      }
    }
    new_node->parent->children.emplace(ckey, new_node);
    child->parent = new_node;
    std::vector<int64_t> child_suffix(child->key.begin() + split_len,
                                      child->key.end());
    PageKey suffix_key = child_key_from(child_suffix);
    new_node->children.emplace(suffix_key, child);

    // fix child key to suffix
    child->key.erase(child->key.begin(), child->key.begin() + split_len);
    return new_node;
  }

  py::tuple insert_helper(TreeNode *node, std::vector<int64_t> &key,
                          torch::Tensor &value) {
    ensure_int64_1d(value);
    TreeNode *new_node = nullptr;
    if (key.empty())
      return py::make_tuple(new_node, 0);

    int64_t device_matched_prefix_len = 0;
    PageKey ckey = child_key_from(key);

    while (!key.empty()) {
      auto it = node->children.find(ckey);
      if (it == node->children.end())
        break;

      node = it->second;
      node->touch();
      int64_t prefix_len = key_match(node->key, key);

      if (prefix_len == node->key.size()) {
        // Full match of node->key
        if (node->evicted()) {
          node->value = value.index({Slice(0, prefix_len)});
          evictable_size_ += node->value.size(0);
        } else {
          node->hit_count += 1;
          device_matched_prefix_len += prefix_len;
        }
      } else {
        // Partial match => split
        TreeNode *new_node = split_node(node, prefix_len);
        if (new_node->evicted()) {
          new_node->value = value.index({Slice(0, prefix_len)});
          evictable_size_ += new_node->value.size(0);
        } else {
          new_node->hit_count += 1;
          device_matched_prefix_len += prefix_len;
        }
        node = new_node;
      }

      // advance by exactly 'prefix_len'
      key.erase(key.begin(), key.begin() + prefix_len);
      value = value.index({Slice(prefix_len, torch::indexing::None)});
      if (!key.empty())
        ckey = child_key_from(key);
    }

    if (!key.empty()) {
      // Create new child with remaining key/value
      new_node = make_node();
      new_node->parent = node;
      new_node->key = key;
      new_node->value = value;
      node->children.emplace(ckey, new_node);
      new_node->hit_count = 1; // first access

      evictable_size_ += value.size(0);
    }
    return py::make_tuple(new_node, device_matched_prefix_len);
  }

  int64_t insert_helper_host(TreeNode *node, std::vector<int64_t> &key,
                             torch::Tensor &host_value,
                             const std::vector<std::string> &hash_value) {
    if (key.empty())
      return 0;

    ensure_int64_1d(host_value);
    node->touch();

    PageKey ckey = child_key_from(key);
    int64_t matched_len = 0;
    while (!key.empty()) {
      auto it = node->children.find(ckey);
      if (it == node->children.end())
        break;

      node = it->second;
      node->touch();
      int64_t prefix_len = key_match(node->key, key);

      key.erase(key.begin(), key.begin() + prefix_len);
      host_value = host_value.index({Slice(prefix_len, torch::indexing::None)});
      matched_len += prefix_len;

      if (prefix_len < node->key.size()) {
        node = split_node(node, prefix_len);
      }
      if (!key.empty())
        ckey = child_key_from(key);
    }

    if (!key.empty()) {
      // Create new child, assign remaining host indices + hashes
      auto *new_node = make_node();
      new_node->parent = node;
      new_node->key = key;
      new_node->host_value = host_value;
      int64_t start_page = matched_len / page_size_;
      int64_t pages_left = key.size() / page_size_;
      new_node->hash_value.assign(hash_value.begin() + start_page,
                                  hash_value.begin() + start_page + pages_left);
      node->children.emplace(child_key_from(key), new_node);
    }
    return matched_len;
  }

  void set_node_value(TreeNode *node, torch::Tensor v) {
    ensure_int64_1d(v);
    if (node->evicted() && node->lock_ref == 0) {
      evictable_size_ += v.size(0);
    }
    node->value = v;
  }

  // Accessors exposed to Python
  TreeNode *root() const { return root_; }
  int page_size() const { return page_size_; }

private:
  int page_size_;
  torch::Device device_;
  TreeNode *root_ = nullptr;
  int64_t next_id_ = 0;

  std::unordered_map<int64_t, std::unique_ptr<TreeNode>> nodes_;
  int64_t evictable_size_ = 0;
  int64_t protected_size_ = 0;

  // ----- internals -----
  TreeNode *make_node() {
    const int64_t id = next_id_++;
    nodes_.emplace(id, std::make_unique<TreeNode>());
    TreeNode *n = nodes_[id].get();
    n->id = id;
    n->last_access_time = now_monotonic();
    return n;
  }

  std::vector<int64_t> page_align(const std::vector<int64_t> &key) const {
    if (page_size_ == 1)
      return key;
    int64_t pages = key.size() / page_size_;
    return std::vector<int64_t>(key.begin(), key.begin() + pages * page_size_);
  }

  size_t key_match(const std::vector<int64_t> &a,
                   const std::vector<int64_t> &b) const {
    if (page_size_ == 1) {
      size_t i = 0, n = std::min(a.size(), b.size());
      while (i < n && a[i] == b[i])
        ++i;
      return i;
    }
    // page-wise
    size_t i = 0;
    size_t n = std::min(a.size(), b.size());
    while (i + page_size_ <= n) {
      bool eq = true;
      for (int k = 0; k < page_size_; ++k) {
        if (a[i + k] != b[i + k]) {
          eq = false;
          break;
        }
      }
      if (!eq)
        break;
      i += page_size_;
    }
    return i;
  }

  PageKey child_key_from(const std::vector<int64_t> &key) const {
    if (page_size_ == 1)
      return PageKey(&key[0], 1);
    return PageKey(&key[0], page_size_);
  }

  std::pair<std::vector<torch::Tensor>, TreeNode *>
  match_prefix_helper(TreeNode *node, const std::vector<int64_t> &key) {
    std::vector<torch::Tensor> vals;
    if (key.empty()) {
      node->touch();
      return {vals, node};
    }

    std::vector<int64_t> rem = key;
    PageKey ckey = child_key_from(rem);
    node->touch();

    while (!rem.empty()) {
      auto it = node->children.find(ckey);
      if (it == node->children.end())
        break;

      TreeNode *child = it->second;
      child->touch();
      size_t prefix = key_match(child->key, rem);
      if (prefix < child->key.size()) {
        TreeNode *new_node = split_node(child, prefix);
        if (!new_node->evicted())
          vals.push_back(new_node->value);
        node = new_node;
        break;
      } else {
        if (!child->evicted())
          vals.push_back(child->value);
        node = child;
        rem.erase(rem.begin(), rem.begin() + prefix);
        if (!rem.empty())
          ckey = child_key_from(rem);
      }
    }
    return {vals, node};
  }

  // Collect device leaves (leaf = has device value and all children evicted)
  std::vector<TreeNode *> collect_leaves_device() const {
    std::vector<TreeNode *> ret_list;
    std::vector<TreeNode *> stack{root_};
    while (!stack.empty()) {
      TreeNode *n = stack.back();
      stack.pop_back();
      if (n->is_leaf_device()) {
        ret_list.push_back(n);
      } else {
        for (auto &kv : n->children) {
          if (kv.second && !kv.second->evicted())
            stack.push_back(kv.second);
        }
      }
    }
    return ret_list;
  }

  // Collect *all* leaves regardless of device state (used by host eviction)
  std::vector<TreeNode *> collect_leaves_all() const {
    std::vector<TreeNode *> ret_list;
    std::vector<TreeNode *> stack{root_};
    while (!stack.empty()) {
      TreeNode *n = stack.back();
      stack.pop_back();
      if (n->is_leaf()) { // only non-locked leaves
        ret_list.push_back(n);
      } else {
        for (auto &kv : n->children)
          stack.push_back(kv.second);
      }
    }
    return ret_list;
  }

  void delete_leaf(TreeNode *leaf) {
    if (!leaf || leaf->is_root())
      return;

    for (auto it = leaf->parent->children.begin();
         it != leaf->parent->children.end(); ++it) {
      if (it->second == leaf) {
        leaf->parent->children.erase(it);
        break;
      }
    }

    nodes_.erase(leaf->id); // remove from ownership map
  }
};

// ---------- pybind ----------
PYBIND11_MODULE(hiradix_core, m) {
  py::class_<TreeNode>(m, "TreeNode")
      .def_readonly("id", &TreeNode::id)
      .def_readwrite("key", &TreeNode::key)
      .def_readwrite("hit_count", &TreeNode::hit_count)
      .def_readwrite("lock_ref", &TreeNode::lock_ref)
      .def_readwrite("host_ref_counter", &TreeNode::host_ref_counter)
      .def_property_readonly("evicted",
                             [](const TreeNode &n) { return n.evicted(); })
      .def_property_readonly("backuped",
                             [](const TreeNode &n) { return n.backuped(); })
      .def_property_readonly("value", [](TreeNode &n) { return n.value; })
      .def_property(
          "host_value", [](TreeNode &n) { return n.host_value; },
          [](TreeNode &n, const torch::Tensor &v) {
            ensure_int64_1d(v);
            n.host_value = v;
          })
      .def_property(
          "backuped_storage", [](TreeNode &n) { return n.backuped_storage; },
          [](TreeNode &n, bool v) { n.backuped_storage = v; })
      .def_property(
          "hash_value", [](TreeNode &n) { return n.hash_value; },
          [](TreeNode &n, const std::vector<std::string> &v) {
            n.hash_value = v;
          })
      .def_property_readonly(
          "parent", [](TreeNode &n) { return n.parent; },
          py::return_value_policy::reference)
      .def("protect_host", &TreeNode::protect_host)
      .def("release_host", &TreeNode::release_host)
      .def("get_last_hash_value", &TreeNode::get_last_hash_value);

  py::class_<HiRadixTreeCore>(m, "HiRadixTreeCore")
      .def(py::init<int, torch::Device>())
      .def("reset", &HiRadixTreeCore::reset)
      .def("match_prefix", &HiRadixTreeCore::match_prefix,
           py::return_value_policy::reference)
      .def("insert_helper", &HiRadixTreeCore::insert_helper)
      .def("insert_helper_host", &HiRadixTreeCore::insert_helper_host)
      .def("inc_lock_ref", &HiRadixTreeCore::inc_lock_ref)
      .def("dec_lock_ref", &HiRadixTreeCore::dec_lock_ref)
      .def("evictable_size", &HiRadixTreeCore::evictable_size)
      .def("protected_size", &HiRadixTreeCore::protected_size)
      .def("set_node_value", &HiRadixTreeCore::set_node_value)
      .def("evict_device", &HiRadixTreeCore::evict_device)
      .def("evict_host", &HiRadixTreeCore::evict_host)
      .def("split_node", &HiRadixTreeCore::split_node)
      .def_property_readonly("root_node", &HiRadixTreeCore::root,
                             py::return_value_policy::reference)
      .def_property_readonly("page_size", &HiRadixTreeCore::page_size);
}
