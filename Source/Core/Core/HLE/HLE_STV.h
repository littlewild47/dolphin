// STV_GC_SKINHLE_v1: reemplazos nativos de funciones de juegos (ver HLE_STV.cpp).
#pragma once

#include <atomic>
#include <cstddef>

#include "Common/CommonTypes.h"

namespace Core
{
class CPUThreadGuard;
class System;
}  // namespace Core

namespace HLE_STV
{
// The Simpsons: Hit & Run (GHQE7D): skinning de vertices por CPU.
constexpr u32 SKIN_DIR = 0x8027d508;
bool FirmaSkinOK(Core::System& system);
void SkinReemplazo(const Core::CPUThreadGuard& guard);
void SkinControl(const Core::CPUThreadGuard& guard);
void ControlComparar(const u8* bytes, size_t n);
extern bool g_skin_control_activo;
extern std::atomic<u32> g_skin_gqr4, g_skin_gqr7, g_skin_directas;
extern std::atomic<u32> g_skin_llamadas, g_skin_vertices, g_skin_comparados, g_skin_difieren, g_skin_primera_dif;
}  // namespace HLE_STV
