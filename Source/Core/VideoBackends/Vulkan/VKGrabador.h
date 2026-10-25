// Copyright 2026 STV
// SPDX-License-Identifier: GPL-2.0-or-later

// STV_GC_GRABADOR_v1: hilo grabador de comandos de Vulkan.
//
// En la consola STV (8 Cortex-A55, Mali-G57) el hilo de video de Dolphin esta en el camino critico, y ~16 % de su
// tiempo en CPU es el driver de la Mali grabando comandos (vkCmd*, vkUpdateDescriptorSets). El grabador saca ese
// trabajo del hilo de video: las llamadas que graban se interceptan en los punteros de funcion globales (los que
// carga LoadVulkanDeviceFunctions; VMA usa los mismos), y desde el hilo de video se copian con sus argumentos a un
// anillo de un productor y un consumidor; el hilo grabador las ejecuta en orden contra el driver.
//
// Reglas:
//  - Solo difiere el hilo de video (Fifo::RunGpuLoop lo anota en Fifo::g_stv_hilo_gpu). Cualquier otro hilo que
//    grabe vacia el anillo antes y llama directo (inicio, cierre, modo de un nucleo).
//  - Lo que depende de que lo grabado ya este en el driver vacia el anillo antes, en el hilo de video: enviar sin
//    el hilo de envio, esperar a que el hilo de envio quede ocioso, adquirir o destruir la cadena, destruir
//    objetos que un comando pendiente pueda usar (salvo en la limpieza diferida, que corre despues de la valla).
//  - El envio por el hilo de envio (BackendMultithreading) se encola EN el anillo: lo empuja el grabador, despues
//    del vkEndCommandBuffer del mismo bufer.
//  - debug.stv.gc.grabador = 0 lo apaga en caliente (se relee cada 30 cuadros; al apagarse vacia).

#pragma once

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

#include "Common/CommonTypes.h"

namespace Vulkan
{
namespace Grabador
{
// Engancha los punteros de Vulkan y crea el hilo. Llamar con las funciones de dispositivo recien cargadas.
void Iniciar();
// Apaga, vacia, desengancha y termina el hilo.
void Terminar();

// true si la llamada actual se difiere (hilo de video + grabador activo).
bool Difiere();
// Espera a que el grabador ejecute todo lo encolado. Sin efecto si no hay nada pendiente.
void Vaciar();
// Releer debug.stv.gc.grabador (en el hilo de video, entre cuadros). Al apagarse, vacia.
void Releer();

// Reserva lugar para una entrada con `bytes` de carga y devuelve el puntero a la carga; Publicar() la entrega.
u8* Reservar(void (*fn)(u8*), size_t bytes);
void Publicar();

// Encola un objeto invocable (que no apunte fuera de si mismo).
template <typename F>
void Encolar(F&& f)
{
  using T = std::decay_t<F>;
  u8* p = Reservar(
      [](u8* c) {
        T* o = reinterpret_cast<T*>(c);
        (*o)();
        o->~T();
      },
      sizeof(T));
  new (p) T(std::forward<F>(f));
  Publicar();
}

// Limpieza diferida en curso (la escribe y la lee solo el hilo de video): lo que se destruye ahi ya no lo usa
// ningun comando pendiente, porque la limpieza corre despues de esperar la valla del bufer.
extern bool g_limpieza;
}  // namespace Grabador
}  // namespace Vulkan
