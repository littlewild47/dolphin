// Copyright 2016 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoBackends/Vulkan/StagingBuffer.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#endif

#include <cstring>

#include "Common/Assert.h"

#include "VideoBackends/Vulkan/CommandBufferManager.h"
#include "VideoBackends/Vulkan/VulkanContext.h"
#include "VideoCommon/DriverDetails.h"

namespace Vulkan
{
StagingBuffer::StagingBuffer(STAGING_BUFFER_TYPE type, VkBuffer buffer, VmaAllocation alloc,
                             VkDeviceSize size, char* map_ptr)
    : m_type(type), m_buffer(buffer), m_alloc(alloc), m_size(size), m_map_pointer(map_ptr)
{
}

StagingBuffer::~StagingBuffer()
{
  // Unmap before destroying
  if (m_map_pointer)
    Unmap();

  g_command_buffer_mgr->DeferBufferDestruction(m_buffer, m_alloc);
}

void StagingBuffer::BufferMemoryBarrier(VkCommandBuffer command_buffer, VkBuffer buffer,
                                        VkAccessFlags src_access_mask,
                                        VkAccessFlags dst_access_mask, VkDeviceSize offset,
                                        VkDeviceSize size, VkPipelineStageFlags src_stage_mask,
                                        VkPipelineStageFlags dst_stage_mask)
{
  VkBufferMemoryBarrier buffer_info = {
      VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,  // VkStructureType    sType
      nullptr,                                  // const void*        pNext
      src_access_mask,                          // VkAccessFlags      srcAccessMask
      dst_access_mask,                          // VkAccessFlags      dstAccessMask
      VK_QUEUE_FAMILY_IGNORED,                  // uint32_t           srcQueueFamilyIndex
      VK_QUEUE_FAMILY_IGNORED,                  // uint32_t           dstQueueFamilyIndex
      buffer,                                   // VkBuffer           buffer
      offset,                                   // VkDeviceSize       offset
      size                                      // VkDeviceSize       size
  };

  vkCmdPipelineBarrier(command_buffer, src_stage_mask, dst_stage_mask, 0, 0, nullptr, 1,
                       &buffer_info, 0, nullptr);
}

bool StagingBuffer::Map()
{
  // The staging buffer is permanently mapped and VMA handles the mapping for us
  return true;
}

void StagingBuffer::Unmap()
{
  // The staging buffer is permanently mapped and VMA handles the unmapping for us
}

void StagingBuffer::FlushCPUCache(VkDeviceSize offset, VkDeviceSize size)
{
  // vmaFlushAllocation checks whether the allocation uses a coherent memory type internally
  vmaFlushAllocation(g_vulkan_context->GetMemoryAllocator(), m_alloc, offset, size);
}

void StagingBuffer::InvalidateGPUCache(VkCommandBuffer command_buffer,
                                       VkAccessFlagBits dest_access_flags,
                                       VkPipelineStageFlagBits dest_pipeline_stage,
                                       VkDeviceSize offset, VkDeviceSize size)
{
  ASSERT((offset + size) <= m_size || (offset < m_size && size == VK_WHOLE_SIZE));
  BufferMemoryBarrier(command_buffer, m_buffer, VK_ACCESS_HOST_WRITE_BIT, dest_access_flags, offset,
                      size, VK_PIPELINE_STAGE_HOST_BIT, dest_pipeline_stage);
}

void StagingBuffer::PrepareForGPUWrite(VkCommandBuffer command_buffer,
                                       VkAccessFlagBits dst_access_flags,
                                       VkPipelineStageFlagBits dst_pipeline_stage,
                                       VkDeviceSize offset, VkDeviceSize size)
{
  ASSERT((offset + size) <= m_size || (offset < m_size && size == VK_WHOLE_SIZE));
  BufferMemoryBarrier(command_buffer, m_buffer, VK_ACCESS_MEMORY_WRITE_BIT, dst_access_flags,
                      offset, size, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, dst_pipeline_stage);
}

void StagingBuffer::FlushGPUCache(VkCommandBuffer command_buffer, VkAccessFlagBits src_access_flags,
                                  VkPipelineStageFlagBits src_pipeline_stage, VkDeviceSize offset,
                                  VkDeviceSize size)
{
  ASSERT((offset + size) <= m_size || (offset < m_size && size == VK_WHOLE_SIZE));
  BufferMemoryBarrier(command_buffer, m_buffer, src_access_flags, VK_ACCESS_HOST_READ_BIT, offset,
                      size, src_pipeline_stage, VK_PIPELINE_STAGE_HOST_BIT);
}

void StagingBuffer::InvalidateCPUCache(VkDeviceSize offset, VkDeviceSize size)
{
  // vmaInvalidateAllocation checks whether the allocation uses a coherent memory type internally
  vmaInvalidateAllocation(g_vulkan_context->GetMemoryAllocator(), m_alloc, offset, size);
}

void StagingBuffer::Read(VkDeviceSize offset, void* data, size_t size, bool invalidate_caches)
{
  ASSERT((offset + size) <= m_size);
  if (invalidate_caches)
    InvalidateCPUCache(offset, size);

  memcpy(data, m_map_pointer + offset, size);
}

void StagingBuffer::Write(VkDeviceSize offset, const void* data, size_t size,
                          bool invalidate_caches)
{
  ASSERT((offset + size) <= m_size);

  memcpy(m_map_pointer + offset, data, size);
  if (invalidate_caches)
    FlushCPUCache(offset, size);
}

bool StagingBuffer::AllocateBuffer(STAGING_BUFFER_TYPE type, VkDeviceSize size,
                                   VkBufferUsageFlags usage, VkBuffer* out_buffer,
                                   VmaAllocation* out_alloc, char** out_map_ptr)
{
  VkBufferCreateInfo buffer_create_info = {
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,  // VkStructureType        sType
      nullptr,                               // const void*            pNext
      0,                                     // VkBufferCreateFlags    flags
      size,                                  // VkDeviceSize           size
      usage,                                 // VkBufferUsageFlags     usage
      VK_SHARING_MODE_EXCLUSIVE,             // VkSharingMode          sharingMode
      0,                                     // uint32_t               queueFamilyIndexCount
      nullptr                                // const uint32_t*        pQueueFamilyIndices
  };

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags =
      VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
  alloc_create_info.pool = VK_NULL_HANDLE;
  alloc_create_info.pUserData = nullptr;
  alloc_create_info.priority = 0.0;
  alloc_create_info.preferredFlags = 0;
  alloc_create_info.requiredFlags = 0;

  // STV_GC_RBCACHE_v1: BUG_SLOW_CACHED_READBACK_MEMORY obliga en toda GPU ARM a leer de memoria coherente SIN cache
  // (tipo 0 en la Mali-G57 de la consola). Leer con memcpy de ahi es lentisimo: las copias del EFB a la RAM (1,3 MB
  // por copia en Wind Waker) eran ~8 % del hilo de video, en la cola de cada cuadro que el hilo de CPU espera. En esta
  // Mali la memoria con cache (tipo 1, no coherente) es la correcta para leer: InvalidateCPUCache (lecturas) y
  // FlushCPUCache (texturas intermedias mutables) ya hacen el mantenimiento. Wind Waker 22,9 -> 24,2 FPS (3 de 3).
  // Solo los buferes de LECTURA; debug.stv.gc.rbcache=0 (leida al crear el bufer) vuelve a la regla de ARM.
  bool stv_rbcache = type == STAGING_BUFFER_TYPE_READBACK;
#ifdef __ANDROID__
  {
    char rb[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.stv.gc.rbcache", rb) > 0 && rb[0] == '0')
      stv_rbcache = false;
  }
#endif
  if (DriverDetails::HasBug(DriverDetails::BUG_SLOW_CACHED_READBACK_MEMORY) && !stv_rbcache) [[unlikely]]
  {
    // If there is no memory type that is both CACHED and COHERENT,
    // pick the one that is COHERENT
    alloc_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
    alloc_create_info.requiredFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  else
  {
    if (type == STAGING_BUFFER_TYPE_UPLOAD)
      alloc_create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    else
      alloc_create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  }

  VmaAllocationInfo alloc_info;
  VkResult res = vmaCreateBuffer(g_vulkan_context->GetMemoryAllocator(), &buffer_create_info,
                                 &alloc_create_info, out_buffer, out_alloc, &alloc_info);

  if (type == STAGING_BUFFER_TYPE_UPLOAD)
  {
    VkMemoryPropertyFlags flags = 0;
    vmaGetMemoryTypeProperties(g_vulkan_context->GetMemoryAllocator(), alloc_info.memoryType,
                               &flags);
    if (!(flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
    {
      WARN_LOG_FMT(VIDEO, "Vulkan: Failed to find a coherent memory type for uploads, this will "
                          "affect performance.");
    }
  }

  *out_map_ptr = static_cast<char*>(alloc_info.pMappedData);

  if (res != VK_SUCCESS)
  {
    LOG_VULKAN_ERROR(res, "vmaCreateBuffer failed: ");
    return false;
  }

  VkMemoryPropertyFlags flags = 0;
  vmaGetAllocationMemoryProperties(g_vulkan_context->GetMemoryAllocator(), *out_alloc, &flags);
#ifdef __ANDROID__
  // STV_GC_RBCACHE_v1: que tipo de memoria toco de verdad (instrumento; pocas lineas por partida).
  if (type == STAGING_BUFFER_TYPE_READBACK)
  {
    __android_log_print(ANDROID_LOG_INFO, "STVGC",
                        "staging lectura: %llu B tipo=%u cached=%d coherent=%d device=%d (rbcache=%d)",
                        static_cast<unsigned long long>(size), alloc_info.memoryType,
                        (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 1 : 0,
                        (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 1 : 0,
                        (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 1 : 0, stv_rbcache ? 1 : 0);
  }
#endif
  return true;
}

std::unique_ptr<StagingBuffer> StagingBuffer::Create(STAGING_BUFFER_TYPE type, VkDeviceSize size,
                                                     VkBufferUsageFlags usage)
{
  VkBuffer buffer;
  VmaAllocation alloc;
  char* map_ptr;
  if (!AllocateBuffer(type, size, usage, &buffer, &alloc, &map_ptr))
    return nullptr;

  return std::make_unique<StagingBuffer>(type, buffer, alloc, size, map_ptr);
}

}  // namespace Vulkan
