#ifndef CUBSORT_H
#define CUBSORT_H

#include <cub/cub.cuh>

template <int BlockThreads, class KeyT>
struct policy_t {
  constexpr static int BLOCK_THREADS = BlockThreads;
  constexpr static int RADIX_BITS = sizeof(KeyT) > 1 ? 6 : 4;

  using LargeSegmentPolicy =
    cub::AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
                                       9,
                                       KeyT,
                                       cub::BLOCK_LOAD_TRANSPOSE,
                                       cub::LOAD_DEFAULT,
                                       cub::RADIX_RANK_MEMOIZE,
                                       cub::BLOCK_SCAN_WARP_SCANS,
                                       RADIX_BITS>;

  using SmallAndMediumSegmentedSortPolicyT =
    cub::AgentSmallAndMediumSegmentedSortPolicy<

      BLOCK_THREADS,

      // Small policy
      cub::AgentSubWarpMergeSortPolicy<4, // threads per problem
                                       5, // items per thread
                                       cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT,
                                       cub::CacheLoadModifier::LOAD_DEFAULT>,

      // Medium policy
      cub::AgentSubWarpMergeSortPolicy<32, // threads per problem
                                       5,  // items per thread
                                       cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT,
                                       cub::CacheLoadModifier::LOAD_DEFAULT>>;
};

template <int BlockThreads, class KeyT, class OffsetT>
inline __device__ void 
cta_sort(OffsetT num_items, cub::detail::device_double_buffer<KeyT> &keys) {
  if (num_items <= 0) {
    return;
  }

  using value_t = cub::NullType;
  using policy = policy_t<BlockThreads, KeyT>;
  using large_policy = typename policy::LargeSegmentPolicy;
  using medium_policy = typename policy::SmallAndMediumSegmentedSortPolicyT::MediumPolicyT;

  constexpr int radix_bits = large_policy::RADIX_BITS;
  constexpr bool is_descending = false;

  using warp_reduce_t = cub::WarpReduce<KeyT>;
  using agent_warp_merge_sort_t = cub::AgentSubWarpSort<is_descending, medium_policy, KeyT, value_t, OffsetT>;
  using agent_segmented_radix_sort_t = cub::AgentSegmentedRadixSort<is_descending, large_policy, KeyT, value_t, OffsetT>;

  __shared__ union {
    typename agent_segmented_radix_sort_t::TempStorage block_sort;
    typename agent_warp_merge_sort_t ::TempStorage medium_warp_sort;
  } temp_storage;

  agent_segmented_radix_sort_t agent(num_items, temp_storage.block_sort);

  constexpr int begin_bit = 0;
  constexpr int end_bit = sizeof(KeyT) * 8;
  constexpr int cacheable_tile_size = BlockThreads * large_policy::ITEMS_PER_THREAD;

  value_t *value_ptr = nullptr;

  if (num_items <= medium_policy::ITEMS_PER_TILE) {
    if (threadIdx.x < medium_policy::WARP_THREADS) {
      agent_warp_merge_sort_t(temp_storage.medium_warp_sort)
          .ProcessSegment(num_items, keys.current(), keys.alternate(), value_ptr, value_ptr);
      keys.swap();
    }
  } else if (num_items < cacheable_tile_size) {
    agent.ProcessSinglePass(begin_bit, end_bit, keys.current(), value_ptr, keys.alternate(), value_ptr);
    keys.swap();
  } else {
    int current_bit = begin_bit;
    int pass_bits = (cub::min)(int{radix_bits}, (end_bit - current_bit));
    agent.ProcessIterative(current_bit, pass_bits, keys.current(), value_ptr, keys.alternate(), value_ptr);
    keys.swap();

    current_bit += pass_bits;
    #pragma unroll 1
    while (current_bit < end_bit) {
      pass_bits = (cub::min)(int{radix_bits}, (end_bit - current_bit));
      cub::CTA_SYNC();
      agent.ProcessIterative(current_bit, pass_bits, keys.current(), value_ptr, keys.alternate(), value_ptr);
      keys.swap();
      current_bit += pass_bits;
    }
  }
}

template <int BlockThreads>
__global__ void kernel(int num_items, int *src, int *buffer, int** result) {
  cub::detail::device_double_buffer<int> keys(src, buffer);
  cta_sort<BlockThreads>(num_items, keys);
  *result = keys.current();
}

#endif