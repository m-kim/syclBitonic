// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <sycl/sycl.hpp>
#include "common.h"

namespace cubit {
  constexpr int BLOCKSIZE=512;

  /*! helper function to compute block count for given number of
      thread and block size; by dividing and rounding up */
  inline int divRoundUp(int a, int b) { return (a+b-1)/b; }

  template<typename T>
  inline T max_value();

  template<>
  inline uint32_t max_value() { return UINT_MAX; };
  template<>
  inline int32_t max_value() { return INT_MAX; };
  template<>
  inline uint64_t max_value() { return ULONG_MAX; };
  template<>
  inline float max_value() { return INFINITY; };
  template<>
  inline double max_value() { return INFINITY; };
  


  template<typename key_t>
  inline static void shm_sort(key_t *const __restrict__ keys,
                                         uint32_t a,
                                         uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    keys[a] = (key_a<key_b)?key_a:key_b;
    keys[b] = (key_a<key_b)?key_b:key_a;
  }

  template<typename key_t, typename val_t>
  inline static void shm_sort(key_t *const __restrict__ keys,
                                         val_t *const __restrict__ vals,
                                         uint32_t a,
                                         uint32_t b, 
                                         const sycl::nd_item<3> &item_ct1)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    val_t val_a = vals[a];
    val_t val_b = vals[b];
    keys[a] = (key_a<key_b)?key_a:key_b;
    keys[b] = (key_a<key_b)?key_b:key_a;
    vals[a] = (key_a<key_b)?val_a:val_b;
    vals[b] = (key_a<key_b)?val_b:val_a;
  }

  template<typename key_t>
  inline static void gmem_sort(key_t *const keys,
                                          uint32_t a,
                                          uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    if (key_b < key_a) {
      keys[a] = key_b;
      keys[b] = key_a;
    }
  }
  
  template<typename key_t, typename val_t>
  inline static void gmem_sort(key_t *const keys,
                                          val_t *const vals,
                                          uint32_t a,
                                          uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    if (key_b < key_a) {
      keys[a] = key_b;
      keys[b] = key_a;

      val_t val_a = vals[a];
      val_t val_b = vals[b];
      vals[a] = val_b;
      vals[b] = val_a;
    }
  }
  

  template<typename key_t>
  void block_sort_up(key_t *const __restrict__ g_keys, uint32_t _N,
                     const sycl::nd_item<3> &item_ct1, key_t *keys)
  {

    uint32_t blockStart = item_ct1.get_group(2) * (2 * BLOCKSIZE);
    if (blockStart + item_ct1.get_local_id(2) < _N)
      keys[item_ct1.get_local_id(2)] =
          g_keys[blockStart + item_ct1.get_local_id(2)];
    else
      keys[item_ct1.get_local_id(2)] = max_value<key_t>();
    if (BLOCKSIZE + blockStart + item_ct1.get_local_id(2) < _N)
      keys[BLOCKSIZE + item_ct1.get_local_id(2)] =
          g_keys[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)];
    else
      keys[BLOCKSIZE + item_ct1.get_local_id(2)] = max_value<key_t>();
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int l, r, s;
    // ======== seq size 1 ==========
    {
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 2 ==========
    {
      s = (int)item_ct1.get_local_id(2) & -2;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (4-1);
      shm_sort(keys,l,r);
      
      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 4 ==========
    {
      s = (int)item_ct1.get_local_id(2) & -4;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (8-1);
      shm_sort(keys,l,r);
      
      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 8 ==========
    {
      s = (int)item_ct1.get_local_id(2) & -8;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (16-1);
      shm_sort(keys,l,r);
      
      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 16 ==========
    {
      // __syncthreads();
      s = (int)item_ct1.get_local_id(2) & -16;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (32-1);
      shm_sort(keys,l,r);
      
      // ------ down seq size 8 ---------
      // __syncthreads();
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 32 ==========
    {
      /*
      DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -32;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (64-1);
      shm_sort(keys,l,r);
      /*
      DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 64 ==========
    {
      /*
      DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -64;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (128-1);
      shm_sort(keys,l,r);
      /*
      DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      /*
      DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 128 ==========
    {
      /*
      DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -128;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (256-1);
      shm_sort(keys,l,r);
      /*
      DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 64 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -64);
      r = l + 64;
      shm_sort(keys,l,r);
      /*
      DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      /*
      DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      /*
      DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 256 ==========
    {
      /*
      DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -256;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (512-1);
      shm_sort(keys,l,r);
      /*
      DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 128 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -128);
      r = l + 128;
      shm_sort(keys,l,r);
      /*
      DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 64 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -64);
      r = l + 64;
      shm_sort(keys,l,r);
      /*
      DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      /*
      DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      /*
      DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      /*
      DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 512 ==========
    {
      /*
      DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -512;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (1024-1);
      shm_sort(keys,l,r);
      /*
      DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 256 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -256);
      r = l + 256;
      shm_sort(keys,l,r);
      /*
      DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 128 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -128);
      r = l + 128;
      shm_sort(keys,l,r);
      /*
      DPCT1065:22: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 64 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -64);
      r = l + 64;
      shm_sort(keys,l,r);
      /*
      DPCT1065:23: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      /*
      DPCT1065:24: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      /*
      DPCT1065:25: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }



    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (blockStart + item_ct1.get_local_id(2) < _N)
        g_keys[blockStart + item_ct1.get_local_id(2)] =
            keys[item_ct1.get_local_id(2)];
    if (BLOCKSIZE + blockStart + item_ct1.get_local_id(2) < _N)
        g_keys[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)] =
            keys[BLOCKSIZE + item_ct1.get_local_id(2)];
  }




  template<typename key_t, typename val_t>
  void block_sort_up(key_t *const __restrict__ g_keys,
                                val_t *const __restrict__ g_vals,
                                uint32_t _N,
                                const sycl::nd_item<3> &item_ct1,
                                key_t *keys,
                                val_t *vals)
  {

    uint32_t blockStart = item_ct1.get_group(2) * (2 * BLOCKSIZE);
    if (blockStart + item_ct1.get_local_id(2) < _N) {
      keys[item_ct1.get_local_id(2)] =
          g_keys[blockStart + item_ct1.get_local_id(2)];
      vals[item_ct1.get_local_id(2)] =
          g_vals[blockStart + item_ct1.get_local_id(2)];
    } else
      keys[item_ct1.get_local_id(2)] = max_value<key_t>();
    if (BLOCKSIZE + blockStart + item_ct1.get_local_id(2) < _N) {
      keys[BLOCKSIZE + item_ct1.get_local_id(2)] =
          g_keys[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)];
      vals[BLOCKSIZE + item_ct1.get_local_id(2)] =
          g_vals[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)];
    } else
      keys[BLOCKSIZE + item_ct1.get_local_id(2)] = max_value<key_t>();
    /*
    DPCT1065:33: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int l, r, s;
    // ======== seq size 1 ==========
    {
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    // ======== seq size 2 ==========
    {
      s = (int)item_ct1.get_local_id(2) & -2;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (4-1);
      shm_sort(keys,vals,l,r,item_ct1);
      
      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    // ======== seq size 4 ==========
    {
      s = (int)item_ct1.get_local_id(2) & -4;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (8-1);
      shm_sort(keys,vals,l,r,item_ct1);
      
      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    // ======== seq size 8 ==========
    {
      s = (int)item_ct1.get_local_id(2) & -8;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (16-1);
      shm_sort(keys,vals,l,r,item_ct1);
      
      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    // ======== seq size 16 ==========
    {
      // __syncthreads();
      s = (int)item_ct1.get_local_id(2) & -16;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (32-1);
      shm_sort(keys,vals,l,r,item_ct1);
      
      // ------ down seq size 8 ---------
      // __syncthreads();
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    // ======== seq size 32 ==========
    {
      /*
      DPCT1065:35: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -32;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (64-1);
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:36: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 8 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    // ======== seq size 64 ==========
    {
      /*
      DPCT1065:37: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -64;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (128-1);
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:38: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:39: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 8 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    // ======== seq size 128 ==========
    {
      /*
      DPCT1065:40: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -128;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (256-1);
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:41: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 64 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -64);
      r = l + 64;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:42: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:43: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 8 ---------
      /*
      DPCT1065:44: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    // ======== seq size 256 ==========
    {
      /*
      DPCT1065:45: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -256;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (512-1);
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:46: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 128 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -128);
      r = l + 128;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:47: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 64 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -64);
      r = l + 64;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:48: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:49: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      /*
      DPCT1065:50: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 8 ---------
      /*
      DPCT1065:51: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    // ======== seq size 512 ==========
    {
      /*
      DPCT1065:52: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s = (int)item_ct1.get_local_id(2) & -512;
      l = item_ct1.get_local_id(2) + s;
      r    = l ^ (1024-1);
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:53: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 256 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -256);
      r = l + 256;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 128 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -128);
      r = l + 128;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 64 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -64);
      r = l + 64;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 8 ---------
      /*
      DPCT1065:58: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    /*
    DPCT1065:34: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (blockStart + item_ct1.get_local_id(2) < _N) {
      g_keys[blockStart + item_ct1.get_local_id(2)] =
          keys[item_ct1.get_local_id(2)];
      g_vals[blockStart + item_ct1.get_local_id(2)] =
          vals[item_ct1.get_local_id(2)];
    }
    if (BLOCKSIZE + blockStart + item_ct1.get_local_id(2) < _N) {
      g_keys[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)] =
          keys[BLOCKSIZE + item_ct1.get_local_id(2)];
      g_vals[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)] =
          vals[BLOCKSIZE + item_ct1.get_local_id(2)];
    }
  }

  




  
  template<typename key_t>
  void block_sort_down(key_t *const __restrict__ g_keys,
                                  uint32_t _N,
                                  const sycl::nd_item<3> &item_ct1,
                                  key_t *keys)
  {

    uint32_t blockStart = item_ct1.get_group(2) * (2 * BLOCKSIZE);
    if (blockStart + item_ct1.get_local_id(2) < _N)
      keys[item_ct1.get_local_id(2)] =
          g_keys[blockStart + item_ct1.get_local_id(2)];
    else
      keys[item_ct1.get_local_id(2)] = max_value<key_t>();
    if (BLOCKSIZE + blockStart + item_ct1.get_local_id(2) < _N)
      keys[BLOCKSIZE + item_ct1.get_local_id(2)] =
          g_keys[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)];
    else
      keys[BLOCKSIZE + item_ct1.get_local_id(2)] = max_value<key_t>();
    /*
    DPCT1065:66: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int l, r;
    // ======== seq size 1024 ==========
    {

      // ------ down seq size 512 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -512);
      r = l + 512;
      shm_sort(keys,l,r);
      /*
      DPCT1065:69: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 256 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -256);
      r = l + 256;
      shm_sort(keys,l,r);
      /*
      DPCT1065:70: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 128 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -128);
      r = l + 128;
      shm_sort(keys,l,r);
      /*
      DPCT1065:71: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 64 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -64);
      r = l + 64;
      shm_sort(keys,l,r);
      /*
      DPCT1065:72: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      /*
      DPCT1065:73: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,l,r);
    }

    /*
    DPCT1065:67: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (blockStart + item_ct1.get_local_id(2) < _N)
        g_keys[blockStart + item_ct1.get_local_id(2)] =
            keys[item_ct1.get_local_id(2)];
    if (BLOCKSIZE + blockStart + item_ct1.get_local_id(2) < _N)
        g_keys[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)] =
            keys[BLOCKSIZE + item_ct1.get_local_id(2)];
  }


  template<typename key_t, typename val_t>
  void block_sort_down(key_t *const __restrict__ g_keys,
                                  val_t *const __restrict__ g_vals,
                                  uint32_t _N,
                                  const sycl::nd_item<3> &item_ct1,
                                  key_t *keys,
                                  val_t *vals)
  {

    uint32_t blockStart = item_ct1.get_group(2) * (2 * BLOCKSIZE);
    if (blockStart + item_ct1.get_local_id(2) < _N) {
      keys[item_ct1.get_local_id(2)] =
          g_keys[blockStart + item_ct1.get_local_id(2)];
      vals[item_ct1.get_local_id(2)] =
          g_vals[blockStart + item_ct1.get_local_id(2)];
    } else
      keys[item_ct1.get_local_id(2)] = max_value<key_t>();
    if (BLOCKSIZE + blockStart + item_ct1.get_local_id(2) < _N) {
      keys[BLOCKSIZE + item_ct1.get_local_id(2)] =
          g_keys[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)];
      vals[BLOCKSIZE + item_ct1.get_local_id(2)] =
          g_vals[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)];
    } else
      keys[BLOCKSIZE + item_ct1.get_local_id(2)] = max_value<key_t>();
    /*
    DPCT1065:74: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int l, r;
    // ======== seq size 1024 ==========
    {
      /*
      DPCT1065:76: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 512 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -512);
      r = l + 512;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:77: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 256 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -256);
      r = l + 256;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:78: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 128 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -128);
      r = l + 128;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:79: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 64 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -64);
      r = l + 64;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:80: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 32 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r,item_ct1);
      /*
      DPCT1065:81: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);

      // ------ down seq size 16 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 8 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 4 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 2 ---------
      l = item_ct1.get_local_id(2) + ((int)item_ct1.get_local_id(2) & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r,item_ct1);

      // ------ down seq size 1 ---------
      l = item_ct1.get_local_id(2) + item_ct1.get_local_id(2);
      r = l + 1;
      shm_sort(keys,vals,l,r,item_ct1);
    }

    /*
    DPCT1065:75: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (blockStart + item_ct1.get_local_id(2) < _N) {
      g_keys[blockStart + item_ct1.get_local_id(2)] =
          keys[item_ct1.get_local_id(2)];
      g_vals[blockStart + item_ct1.get_local_id(2)] =
          vals[item_ct1.get_local_id(2)];
    }
    if (BLOCKSIZE + blockStart + item_ct1.get_local_id(2) < _N) {
      g_keys[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)] =
          keys[BLOCKSIZE + item_ct1.get_local_id(2)];
      g_vals[BLOCKSIZE + blockStart + item_ct1.get_local_id(2)] =
          vals[BLOCKSIZE + item_ct1.get_local_id(2)];
    }
  }

  template<typename key_t>
  void big_down(key_t *const __restrict__ keys,
                           uint32_t N,
                            int seqLen,
                            const sycl::nd_item<3> &item_ct1)
  {
    int tid = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);

    int s    = tid & -seqLen;
    int l    = tid+s;
    int r    = l + seqLen;

    if (r < N)
      gmem_sort(keys,l,r);
  }

  template<typename key_t, typename val_t>
  void big_down(key_t *const __restrict__ keys,
                           val_t *const __restrict__ vals,
                           uint32_t N,
                           int seqLen,
                           const sycl::nd_item<3> &item_ct1)
  {
    int tid = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);

    int s    = tid & -seqLen;
    int l    = tid+s;
    int r    = l + seqLen;

    if (r < N)
      gmem_sort(keys,vals,l,r);
  }

  template<typename key_t>
  void big_up(key_t *const __restrict__ keys,
                         uint32_t N,
                         int seqLen,
                         const sycl::nd_item<3> &item_ct1)
  {
    int tid = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (tid >= N) return;
    
    int s    = tid & -seqLen;
    int l    = tid+s;
    int r    = l ^ (2*seqLen-1);

    if (r < N) {
      gmem_sort(keys,l,r);
    }
  }
  template<typename key_t, typename val_t>
  void big_up(key_t *const __restrict__ keys,
                         val_t *const __restrict__ vals,
                         uint32_t N,
                         int seqLen,
                         const sycl::nd_item<3> &item_ct1)
  {
    int tid = item_ct1.get_global_id(2);
    if (tid >= N) return;
    
    int s    = tid & -seqLen;
    int l    = tid+s;
    int r    = l ^ (2*seqLen-1);

    if (r < N) {
      gmem_sort(keys,vals,l,r);
    }
  }

  template <typename key_t>
  inline void sort(key_t *const __restrict__ d_values, size_t numValues,
                   sycl::queue q)

  {
    int bs = 512;
    int numValuesPerBlock = 2*bs;

    // ==================================================================
    // first - sort all blocks of 2x1024 using per-block sort
    // ==================================================================
    int nb = divRoundUp((int)numValues,numValuesPerBlock);
    /*
    DPCT1049:82: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
   {
      //dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
      q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<key_t, 1> keys_acc_ct1(sycl::range<1>(2 * BLOCKSIZE),
                                                     cgh);

         cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                sycl::range<3>(1, 1, bs),
                                            sycl::range<3>(1, 1, bs)),
                          [=](sycl::nd_item<3> item_ct1) {
                             block_sort_up(d_values, (int)numValues);
                          });
      });
   }

    int _nb = divRoundUp(int(numValues),BLOCKSIZE);
    for (int upLen=numValuesPerBlock;upLen<numValues;upLen+=upLen) {
      /*
      DPCT1049:83: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      q.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, _nb) *
                                                 sycl::range<3>(1, 1, bs),
                                             sycl::range<3>(1, 1, bs)),
                           [=](sycl::nd_item<3> item_ct1) {
                              big_up(d_values, (int)numValues, upLen);
                           });
      for (int downLen=upLen/2;downLen>BLOCKSIZE;downLen/=2) {
        /*
        DPCT1049:84: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
         q.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, _nb) *
                                                    sycl::range<3>(1, 1, bs),
                                                sycl::range<3>(1, 1, bs)),
                              [=](sycl::nd_item<3> item_ct1) {
                                 big_down(d_values, (int)numValues, downLen);
                              });
      }
      /*
      DPCT1049:85: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      {
        //  dpct::has_capability_or_fail(stream->get_device(),
        //                               {sycl::aspect::fp64});
         q.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<key_t, 1> keys_acc_ct1(
                sycl::range<1>(2 * BLOCKSIZE), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, bs),
                                               sycl::range<3>(1, 1, bs)),
                             [=](sycl::nd_item<3> item_ct1) {
                                block_sort_down(d_values, (int)numValues);
                             });
         });
      }
    }
  }

  template <typename key_t, typename val_t>
  inline void sort(key_t *const __restrict__ d_keys,
                   val_t *const __restrict__ d_values, size_t numValues,
                   sycl::queue q)

  {
    int bs = 512;
    const int numValuesPerBlock = 2*bs;
    auto device = q.get_device();
    std::cout << "max_work_group_size: " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    // ==================================================================
    // first - sort all blocks of 2x1024 using per-block sort
    // ==================================================================
    int nb = divRoundUp((int)numValues,numValuesPerBlock);
    /*
    DPCT1049:86: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    // dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<key_t, 1> keys_acc_ct1(sycl::range<1>(2 * BLOCKSIZE),
                                                    cgh);
        sycl::local_accessor<val_t, 1> vals_acc_ct1(sycl::range<1>(2 * BLOCKSIZE),
                                                    cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                              sycl::range<3>(1, 1, bs),
                                          sycl::range<3>(1, 1, bs)),
                        [=](sycl::nd_item<3> item_ct1) {
                            block_sort_up(d_keys, d_values,
                                          (uint32_t)numValues, item_ct1,
                                          (key_t *)keys_acc_ct1.get_pointer(),
                                          (val_t *)vals_acc_ct1.get_pointer());
                        });
    }).wait();


    int _nb = divRoundUp(int(numValues),BLOCKSIZE);
    for (int upLen=numValuesPerBlock;upLen<numValues;upLen+=upLen) {
    //   /*
    //   DPCT1049:87: The work-group size passed to the SYCL kernel may exceed the
    //   limit. To get the device limit, query info::device::max_work_group_size.
    //   Adjust the work-group size if needed.
    //   */
      q.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, _nb) *
                                                 sycl::range<3>(1, 1, bs),
                                             sycl::range<3>(1, 1, bs)),
                           [=](sycl::nd_item<3> item_ct1) {
                              big_up(d_keys, d_values, (uint32_t)numValues,
                                     upLen, item_ct1);
                           }).wait();
      for (int downLen=upLen/2;downLen>BLOCKSIZE;downLen/=2) {
        /*
        DPCT1049:88: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
         q.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, _nb) *
                                                    sycl::range<3>(1, 1, bs),
                                                sycl::range<3>(1, 1, bs)),
                              [=](sycl::nd_item<3> item_ct1) {
                                 big_down(d_keys, d_values, (uint32_t)numValues,
                                          downLen, item_ct1);
                              }).wait();
      }
      /*
      DPCT1049:89: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      //  dpct::has_capability_or_fail(stream->get_device(),
      //                               {sycl::aspect::fp64});
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<key_t, 1> keys_acc_ct1(
            sycl::range<1>(2 * BLOCKSIZE), cgh);
        sycl::local_accessor<val_t, 1> vals_acc_ct1(
            sycl::range<1>(2 * BLOCKSIZE), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                sycl::range<3>(1, 1, bs),
                                            sycl::range<3>(1, 1, bs)),
                          [=](sycl::nd_item<3> item_ct1) {
                            block_sort_down(
                                d_keys, d_values, (uint32_t)numValues,
                                item_ct1,
                                (key_t *)keys_acc_ct1.get_pointer(),
                                (val_t *)vals_acc_ct1.get_pointer());
                          });
      }).wait();
    }
  }
}

