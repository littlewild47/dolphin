// Copyright 2022 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/PerformanceMetrics.h"

#include <algorithm>

#include <imgui.h>
#include <implot.h>

#include "Common/HookableEvent.h"
#include "Core/Config/GraphicsSettings.h"
#include "Core/Core.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VideoConfig.h"
#include "Core/Config/MainSettings.h"
#include "Core/CoreTiming.h"
#include "Core/HW/GPFifo.h"

#ifdef __ANDROID__
// STV_GC_METRICAS_v2: con debug.stv.gc.metricas=1, una linea por segundo en logcat (etiqueta STVGC) con los
// cuadros, los VI y la velocidad de la emulacion; el cuadro mas largo, los de mas de 50 ms, las compilaciones de
// pipelines, la cache de texturas y si el JIT sigue ramas (JITFollowBranch, ajustable por juego). Instrumento del banco de GameCube de STV: no depende de que el
// backend dibuje (backend Null) ni de capturar la pantalla.
#include <android/log.h>
#include <sys/system_properties.h>
#endif

PerformanceMetrics::PerformanceMetrics()
{
  const auto invalidate_counters_last_time = [this](Core::State) {
    m_fps_counter.InvalidateLastTime();
    m_vps_counter.InvalidateLastTime();
  };
  m_state_change_hook = Core::AddOnStateChangedCallback(invalidate_counters_last_time);
}

void PerformanceMetrics::Reset()
{
  m_fps_counter.Reset();
  m_vps_counter.Reset();

  m_time_sleeping = DT::zero();
  m_samples = {};

  m_speed = 0;
  m_max_speed = 0;

  m_frame_presentation_offset = DT{};
}

// STV_GC_METRICAS: cuadro mas largo y cuadros de mas de 50 ms desde la ultima linea (los tirones que el promedio
// por segundo esconde). CountFrame corre en el hilo de video; la linea la escribe el hilo de CPU.
namespace CommandProcessor
{
extern bool g_stv_fifo_rapido;  // STV_GC_FIFO_v1 (CommandProcessor.cpp)
extern bool g_stv_fifo_lote;    // STV_GC_FIFO3_v1
}
static std::atomic<u32> s_stv_max_us{0}, s_stv_n50{0};
extern std::atomic<u32> g_stv_pipes, g_stv_pipes_us;  // ShaderCache.cpp
extern std::atomic<u32> g_stv_tex_n, g_stv_tex_kb, g_stv_pool_n, g_stv_pool_kb;  // TextureCacheBase.cpp
static TimePoint s_stv_ultimo_cuadro{};

void PerformanceMetrics::CountFrame()
{
  m_fps_counter.Count();
  const TimePoint ahora = Clock::now();
  if (s_stv_ultimo_cuadro != TimePoint{})
  {
    const u32 us = static_cast<u32>(
        std::chrono::duration_cast<std::chrono::microseconds>(ahora - s_stv_ultimo_cuadro).count());
    if (us > s_stv_max_us.load(std::memory_order_relaxed))
      s_stv_max_us.store(us, std::memory_order_relaxed);
    if (us > 50000)
      s_stv_n50.fetch_add(1, std::memory_order_relaxed);
  }
  s_stv_ultimo_cuadro = ahora;
}

void PerformanceMetrics::CountVBlank()
{
  m_vps_counter.Count();
}

void PerformanceMetrics::CountThrottleSleep(DT sleep)
{
  m_time_sleeping += sleep;
}

void PerformanceMetrics::AdjustClockSpeed(s64 ticks, u32 new_ppc_clock, u32 old_ppc_clock)
{
  for (auto& sample : m_samples)
  {
    const s64 diff = (sample.core_ticks - ticks) * new_ppc_clock / old_ppc_clock;
    sample.core_ticks = ticks + diff;
  }
}

void PerformanceMetrics::CountPerformanceMarker(s64 core_ticks, u32 ticks_per_second)
{
  const auto clock_time = Clock::now();
  const auto work_time = clock_time - m_time_sleeping;

  m_samples.emplace_back(
      PerfSample{.clock_time = clock_time, .work_time = work_time, .core_ticks = core_ticks});

  const auto sample_window = std::chrono::microseconds{g_ActiveConfig.iPerfSampleUSec};
  while (clock_time - m_samples.front().clock_time > sample_window)
    m_samples.pop_front();

  // Avoid division by zero when we just have one sample.
  if (m_samples.size() < 2)
    return;

  const PerfSample& oldest = m_samples.front();
  const auto elapsed_core_time = DT_s(core_ticks - oldest.core_ticks) / ticks_per_second;

  m_speed.store(elapsed_core_time / (clock_time - oldest.clock_time), std::memory_order_relaxed);
  m_max_speed.store(elapsed_core_time / (work_time - oldest.work_time), std::memory_order_relaxed);

#ifdef __ANDROID__
  static TimePoint stv_ultima{};
  if (clock_time - stv_ultima >= std::chrono::seconds{1})
  {
    stv_ultima = clock_time;
    // Los contadores de la ventana se toman SIEMPRE (con las metricas apagadas tambien): si no, la primera linea
    // despues de prenderlas traia todo lo acumulado (f50=861 tras 60 s apagadas, medido en f237).
    const double stv_fmax = s_stv_max_us.exchange(0) / 1000.0;
    const u32 stv_n50 = s_stv_n50.exchange(0), stv_pipes = g_stv_pipes.exchange(0);
    const double stv_pipems = g_stv_pipes_us.exchange(0) / 1000.0;
    {
      // STV_GC_FIFO_v1: debug.stv.gc.fifo = 0 vuelve al camino original de las rafagas del FIFO.
      char fv[PROP_VALUE_MAX] = {};
      const bool r = !(__system_property_get("debug.stv.gc.fifo", fv) > 0 && fv[0] == '0');
      if (r != CommandProcessor::g_stv_fifo_rapido)
      {
        CommandProcessor::g_stv_fifo_rapido = r;
        __android_log_print(ANDROID_LOG_INFO, "STVGC", "fifo rapido=%d", r ? 1 : 0);
      }
    }
    // STV_GC_JITDC_v1: debug.stv.gc.dc = 0|1 apaga/prende el downcount en W27 del JIT ARM64. Esta en
    // la tabla de ajustes del JIT: el proximo CheckForConfigChanges (dentro de GlobalAdvance) vacia la cache y
    // regenera las rutinas en el mismo ClearCache que aplica el valor nuevo.
    {
      char dc[PROP_VALUE_MAX] = {};
      static int stv_dc_prev = -1;
      if (__system_property_get("debug.stv.gc.dc", dc) > 0 && (dc[0] == '0' || dc[0] == '1'))
      {
        const int n = dc[0] - '0';
        if (n != stv_dc_prev)
        {
          stv_dc_prev = n;
          Config::SetCurrent(Config::MAIN_STV_DOWNCOUNT_REG, n == 1);
          __android_log_print(ANDROID_LOG_INFO, "STVGC", "jit: dc=%d", n);
        }
      }
    }
    // STV_GC_JITGF_v1: debug.stv.gc.gf = 0|1|2 apaga/prende los GPR fijos del JIT ARM64 (2 = con
    // verificacion). Mismo camino que dc: el proximo CheckForConfigChanges vacia la cache. El modo EFECTIVO lo
    // escribe JitBase::RefreshConfig ("jit: gf efectivo=..."): con jo.memcheck queda en 0 aunque se pida 1 o 2.
    {
      char gf[PROP_VALUE_MAX] = {};
      static int stv_gf_prev = -1;
      if (__system_property_get("debug.stv.gc.gf", gf) > 0 && gf[0] >= '0' && gf[0] <= '2')
      {
        const int n = gf[0] - '0';
        if (n != stv_gf_prev)
        {
          stv_gf_prev = n;
          Config::SetCurrent(Config::MAIN_STV_GPR_FIJOS, n);
          __android_log_print(ANDROID_LOG_INFO, "STVGC", "jit: gf=%d", n);
        }
      }
    }
    // STV_GC_BLRFIX_v1: debug.stv.gc.blrfix = 0|1 (tabla de ajustes del JIT: vacia la cache).
    {
      char bf[PROP_VALUE_MAX] = {};
      static int stv_bf_prev = -1;
      if (__system_property_get("debug.stv.gc.blrfix", bf) > 0 && (bf[0] == '0' || bf[0] == '1'))
      {
        const int n = bf[0] - '0';
        if (n != stv_bf_prev)
        {
          stv_bf_prev = n;
          Config::SetCurrent(Config::MAIN_STV_BLR_FIX, n == 1);
          __android_log_print(ANDROID_LOG_INFO, "STVGC", "jit: blrfix=%d", n);
        }
      }
    }
    // STV_GC_GQR_v1: debug.stv.gc.gqr = 0|1 (tabla de ajustes del JIT: vacia la cache).
    {
      char gq[PROP_VALUE_MAX] = {};
      static int stv_gq_prev = -1;
      if (__system_property_get("debug.stv.gc.gqr", gq) > 0 && (gq[0] == '0' || gq[0] == '1'))
      {
        const int n = gq[0] - '0';
        if (n != stv_gq_prev)
        {
          stv_gq_prev = n;
          Config::SetCurrent(Config::MAIN_STV_GQR, n == 1);
          __android_log_print(ANDROID_LOG_INFO, "STVGC", "jit: gqr=%d", n);
        }
      }
    }
    // STV_GC_FIFORUT_v1: debug.stv.gc.fiforut = 0|1 (tabla de ajustes del JIT: vacia la cache y rehace las rutinas).
    {
      char fu[PROP_VALUE_MAX] = {};
      static int stv_fu_prev = -1;
      if (__system_property_get("debug.stv.gc.fiforut", fu) > 0 && (fu[0] == '0' || fu[0] == '1'))
      {
        const int n = fu[0] - '0';
        if (n != stv_fu_prev)
        {
          stv_fu_prev = n;
          Config::SetCurrent(Config::MAIN_STV_FIFO_RUT, n == 1);
          __android_log_print(ANDROID_LOG_INFO, "STVGC", "jit: fiforut=%d", n);
        }
      }
    }
    // STV_GC_CERO_v1: debug.stv.gc.cero = 0|1 (tabla de ajustes del JIT: vacia la cache).
    {
      char ce[PROP_VALUE_MAX] = {};
      static int stv_ce_prev = -1;
      if (__system_property_get("debug.stv.gc.cero", ce) > 0 && (ce[0] == '0' || ce[0] == '1'))
      {
        const int n = ce[0] - '0';
        if (n != stv_ce_prev)
        {
          stv_ce_prev = n;
          Config::SetCurrent(Config::MAIN_STV_CERO, n == 1);
          __android_log_print(ANDROID_LOG_INFO, "STVGC", "jit: cero=%d", n);
        }
      }
    }
    // STV_GC_FIFOREG_v1: debug.stv.gc.fiforeg = 0|1 (tabla de ajustes del JIT: vacia la cache).
    {
      char fr[PROP_VALUE_MAX] = {};
      static int stv_fr_prev = -1;
      if (__system_property_get("debug.stv.gc.fiforeg", fr) > 0 && (fr[0] == '0' || fr[0] == '1'))
      {
        const int n = fr[0] - '0';
        if (n != stv_fr_prev)
        {
          stv_fr_prev = n;
          Config::SetCurrent(Config::MAIN_STV_FIFO_REG, n == 1);
          __android_log_print(ANDROID_LOG_INFO, "STVGC", "jit: fiforeg=%d", n);
        }
      }
    }
    // STV_GC_FIFO3_v1: debug.stv.gc.gpumbral = 32..256 (32 = Dolphin; recompila y fija el lote). Con las metricas,
    // una linea por segundo con los vaciados y las rafagas.
    {
      char gu[PROP_VALUE_MAX] = {};
      static int stv_gu_prev = -1;
      if (__system_property_get("debug.stv.gc.gpumbral", gu) > 0)
      {
        const int n = atoi(gu);
        if (n != stv_gu_prev && n >= 32 && n <= 256)
        {
          stv_gu_prev = n;
          Config::SetCurrent(Config::MAIN_STV_GP_UMBRAL, n);
          __android_log_print(ANDROID_LOG_INFO, "STVGC", "fifo: umbral=%d", n);
        }
      }
      static u64 va0 = 0, ra0 = 0;
      char mv[PROP_VALUE_MAX] = {};
      if (__system_property_get("debug.stv.gc.metricas", mv) > 0 && mv[0] == '1')
        __android_log_print(ANDROID_LOG_INFO, "STVGC", "fifo: vaciados=%llu/s rafagas=%llu/s umbral=%u lote=%d",
                            static_cast<unsigned long long>(GPFifo::g_stv_vaciados - va0),
                            static_cast<unsigned long long>(GPFifo::g_stv_rafagas - ra0),
                            GPFifo::g_stv_gp_umbral_c, CommandProcessor::g_stv_fifo_lote ? 1 : 0);
      va0 = GPFifo::g_stv_vaciados;
      ra0 = GPFifo::g_stv_rafagas;
    }
    // STV_GC_SALTO_x: debug.stv.gc.salto = 0|1 (salto de cuadros adaptativo con el interruptor VISkip, ver CoreTiming::Idle). Con las metricas,
    // una linea por segundo con los lazos ociosos que esperaron al video y los que no.
    {
      char sa[PROP_VALUE_MAX] = {};
      static int stv_sa_prev = -1;
      if (__system_property_get("debug.stv.gc.salto", sa) > 0 && sa[0] >= '0' && sa[0] <= '9')
      {
        const int n = sa[0] - '0';
        if (n != stv_sa_prev)
        {
          stv_sa_prev = n;
          Config::SetCurrent(Config::MAIN_STV_SALTO, n);
          __android_log_print(ANDROID_LOG_INFO, "STVGC", "salto: modo=%d", n);
        }
      }
      static u64 es0 = 0, sl0 = 0;
      static u32 di0 = 0;
      const u32 di = g_stv_dibujos_saltados;
      const u64 es = CoreTiming::g_stv_esperas.load(std::memory_order_relaxed);
      const u64 sl = CoreTiming::g_stv_saltos.load(std::memory_order_relaxed);
      char mv[PROP_VALUE_MAX] = {};
      if (__system_property_get("debug.stv.gc.metricas", mv) > 0 && mv[0] == '1')
        __android_log_print(ANDROID_LOG_INFO, "STVGC", "salto: esperas=%llu/s saltos=%llu/s cuadros_sin_dibujar=%u/s modo=%d visk=%d",
                            static_cast<unsigned long long>(es - es0),
                            static_cast<unsigned long long>(sl - sl0), di - di0,
                            Config::Get(Config::MAIN_STV_SALTO), g_ActiveConfig.bVISkip ? 1 : 0);
      es0 = es;
      sl0 = sl;
      di0 = di;
    }
    char v[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.stv.gc.metricas", v) > 0 && v[0] == '1')
    {
      __android_log_print(ANDROID_LOG_INFO, "STVGC",
                          "fps=%.2f vps=%.2f speed=%.3f max=%.3f fmax=%.1f f50=%u pipes=%u pipems=%.1f "
                          "tex=%u/%uMB pool=%u/%uMB rama=%d",
                          GetFPS(), GetVPS(), GetSpeed(), GetMaxSpeed(), stv_fmax, stv_n50, stv_pipes,
                          stv_pipems,
                          g_stv_tex_n.load(), g_stv_tex_kb.load() / 1024, g_stv_pool_n.load(),
                          g_stv_pool_kb.load() / 1024,
                          Config::Get(Config::MAIN_JIT_FOLLOW_BRANCH) ? 1 : 0);
    }
  }
#endif
}

double PerformanceMetrics::GetFPS() const
{
  return m_fps_counter.GetHzAvg();
}

double PerformanceMetrics::GetVPS() const
{
  return m_vps_counter.GetHzAvg();
}

double PerformanceMetrics::GetSpeed() const
{
  return m_speed.load(std::memory_order_relaxed);
}

double PerformanceMetrics::GetMaxSpeed() const
{
  return m_max_speed.load(std::memory_order_relaxed);
}

void PerformanceMetrics::SetLatestFramePresentationOffset(DT offset)
{
  m_frame_presentation_offset.store(offset, std::memory_order_relaxed);
}

void PerformanceMetrics::SetLatestFrameBufferSize(u32 width, u32 height)
{
  m_frame_buffer_size.store(FrameBufferSize{width, height}, std::memory_order_relaxed);
}

void PerformanceMetrics::DrawImGuiStats(const float backbuffer_scale)
{
  m_vps_counter.UpdateStats();
  m_fps_counter.UpdateStats();

  const bool movable_overlays = Config::Get(Config::GFX_MOVABLE_PERFORMANCE_METRICS);
  const int movable_flag = movable_overlays ? ImGuiWindowFlags_None : ImGuiWindowFlags_NoMove;

  const float bg_alpha = 0.7f;
  const auto imgui_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
                           ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoNav | movable_flag |
                           ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing;

  const double fps = GetFPS();
  const double vps = GetVPS();
  const double speed = GetSpeed();

  static ImVec2 last_display_size(-1.0f, -1.0f);

  // Change Color based on % Speed
  float r = 0.0f, g = 1.0f, b = 1.0f;
  if (g_ActiveConfig.bShowSpeedColors)
  {
    r = 1.0 - (speed - 0.8) / 0.2;
    g = speed / 0.8;
    b = (speed - 0.9) / 0.1;
  }

  const float window_padding = 8.f * backbuffer_scale;

  const ImVec2& display_size = ImGui::GetIO().DisplaySize;
  const bool display_size_changed =
      display_size.x != last_display_size.x || display_size.y != last_display_size.y;
  last_display_size = display_size;
  // There are too many edge cases to reasonably handle when the display size changes, so just reset
  // the layout to default. Hopefully users aren't changing window sizes or resolutions too often.
  const ImGuiCond set_next_position_condition =
      (display_size_changed || !movable_overlays) ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
  // Reset the graph size when changing resolutions, and otherwise let the user manually resize it.
  const ImGuiCond set_next_size_condition =
      display_size_changed ? ImGuiCond_Always : ImGuiCond_FirstUseEver;

  float window_y = window_padding;
  float window_x = display_size.x - window_padding;

  const auto clamp_window_position = [&] {
    const ImVec2 position = ImGui::GetWindowPos();
    const ImVec2 size = ImGui::GetWindowSize();
    const float window_min_x = window_padding;
    const float window_max_x = display_size.x - window_padding - size.x;
    const float window_min_y = window_padding;
    const float window_max_y = display_size.y - window_padding - size.y;

    if (window_min_x > window_max_x || window_min_y > window_max_y)
      return;

    const float clamped_window_x = std::clamp(position.x, window_min_x, window_max_x);
    const float clamped_window_y = std::clamp(position.y, window_min_y, window_max_y);
    const bool window_needs_clamping =
        (clamped_window_x != position.x) || (clamped_window_y != position.y);

    if (window_needs_clamping)
      ImGui::SetWindowPos(ImVec2(clamped_window_x, clamped_window_y), ImGuiCond_Always);
  };

  const float min_auto_graph_width = 200.f * backbuffer_scale + 2.f * window_padding;
  const float min_auto_graph_height = 144.f * backbuffer_scale + 2.f * window_padding;

  const float graph_width = std::max(min_auto_graph_width, display_size.x / 4.f);
  const float graph_height = std::max(min_auto_graph_height, display_size.y / 4.f);

  const bool stack_vertically = !g_ActiveConfig.bShowGraphs;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 14.f * backbuffer_scale);
  if (g_ActiveConfig.bShowGraphs)
  {
    // A font size of 13 is small enough to keep the tick numbers from overlapping too much.
    ImGui::PushFont(nullptr, 13.0f);
    ImGui::PushStyleColor(ImGuiCol_ResizeGrip, 0);
    const auto graph_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoNav | movable_flag |
                             ImGuiWindowFlags_NoFocusOnAppearing;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 4.f * backbuffer_scale));

    // Position in the top-right corner of the screen.
    ImGui::SetNextWindowPos(ImVec2(window_x, window_y), set_next_position_condition,
                            ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(graph_width, graph_height), set_next_size_condition);
    ImGui::SetNextWindowBgAlpha(bg_alpha);
    if (ImGui::Begin("PerformanceGraphs", nullptr, graph_flags))
    {
      static constexpr std::size_t num_ticks = 17;
      static constexpr std::array<double, num_ticks> tick_marks = {0.0,
                                                                   1000.0 / 360.0,
                                                                   1000.0 / 240.0,
                                                                   1000.0 / 180.0,
                                                                   1000.0 / 120.0,
                                                                   1000.0 / 90.00,
                                                                   1000.0 / 59.94,
                                                                   1000.0 / 40.00,
                                                                   1000.0 / 29.97,
                                                                   1000.0 / 24.00,
                                                                   1000.0 / 20.00,
                                                                   1000.0 / 15.00,
                                                                   1000.0 / 10.00,
                                                                   1000.0 / 5.000,
                                                                   1000.0 / 2.000,
                                                                   1000.0,
                                                                   2000.0};

      clamp_window_position();
      window_y += ImGui::GetWindowHeight() + window_padding;

      const DT vblank_time = m_vps_counter.GetDtAvg() + 2 * m_vps_counter.GetDtStd();
      const DT frame_time = m_fps_counter.GetDtAvg() + 2 * m_fps_counter.GetDtStd();
      const double target_max_time = DT_ms(vblank_time + frame_time).count();
      const double a =
          std::max(0.0, 1.0 - std::exp(-4.0 * (DT_s(m_fps_counter.GetLastRawDt()) /
                                               DT_s(m_fps_counter.GetSampleWindow()))));

      if (std::isfinite(m_graph_max_time))
        m_graph_max_time += a * (target_max_time - m_graph_max_time);
      else
        m_graph_max_time = target_max_time;

      const double total_frame_time =
          DT_ms(std::max(m_fps_counter.GetSampleWindow(), m_vps_counter.GetSampleWindow())).count();

      if (ImPlot::BeginPlot("PerformanceGraphs", ImVec2(-1.0, -1.0),
                            ImPlotFlags_NoFrame | ImPlotFlags_NoTitle | ImPlotFlags_NoMenus |
                                ImPlotFlags_NoInputs))
      {
        ImPlot::PushStyleColor(ImPlotCol_PlotBg, {0, 0, 0, 0});
        ImPlot::PushStyleColor(ImPlotCol_LegendBg, {0, 0, 0, 0.2f});
        ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.f, 0.f));
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f * backbuffer_scale);
        ImPlot::SetupAxes(nullptr, nullptr,
                          ImPlotAxisFlags_Lock | ImPlotAxisFlags_Invert |
                              ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_NoHighlight,
                          ImPlotAxisFlags_Lock | ImPlotAxisFlags_Invert | ImPlotAxisFlags_NoLabel |
                              ImPlotAxisFlags_NoHighlight);
        ImPlot::SetupAxisFormat(ImAxis_Y1, "%.1f");
        ImPlot::SetupAxisTicks(ImAxis_Y1, tick_marks.data(), num_ticks);
        ImPlot::SetupAxesLimits(0, total_frame_time, 0, m_graph_max_time, ImGuiCond_Always);
        ImPlot::SetupLegend(ImPlotLocation_SouthEast, ImPlotLegendFlags_None);
        m_vps_counter.ImPlotPlotLines("V-Blank (ms)");
        m_fps_counter.ImPlotPlotLines("Frame (ms)");
        ImPlot::EndPlot();
        ImPlot::PopStyleVar(2);
        ImPlot::PopStyleColor(2);
      }
      ImGui::PopStyleVar();
    }
    ImGui::End();
    ImGui::PopFont();
    ImGui::PopStyleColor();
  }

  if (g_ActiveConfig.bShowSpeed)
  {
    // Position in the top-right corner of the screen.
    ImGui::SetNextWindowPos(ImVec2(window_x, window_y), set_next_position_condition,
                            ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(bg_alpha);

    if (ImGui::Begin("SpeedStats", nullptr, imgui_flags))
    {
      if (stack_vertically)
        window_y += ImGui::GetWindowHeight() + window_padding;
      else
        window_x -= ImGui::GetWindowWidth() + window_padding;
      clamp_window_position();
      ImGui::TextColored(ImVec4(r, g, b, 1.0f), "Speed:%4.0lf%%", 100.0 * speed);
      ImGui::TextColored(ImVec4(r, g, b, 1.0f), "Max:%6.0lf%%", 100.0 * GetMaxSpeed());
    }
    ImGui::End();
  }

  if (g_ActiveConfig.bShowFPS || g_ActiveConfig.bShowFTimes)
  {
    // Position in the top-right corner of the screen.
    ImGui::SetNextWindowPos(ImVec2(window_x, window_y), set_next_position_condition,
                            ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(bg_alpha);

    if (ImGui::Begin("FPSStats", nullptr, imgui_flags))
    {
      if (stack_vertically)
        window_y += ImGui::GetWindowHeight() + window_padding;
      else
        window_x -= ImGui::GetWindowWidth() + window_padding;
      clamp_window_position();
      if (g_ActiveConfig.bShowFPS)
        ImGui::TextColored(ImVec4(r, g, b, 1.0f), "FPS:%7.2lf", fps);
      if (g_ActiveConfig.bShowFTimes)
      {
        ImGui::TextColored(ImVec4(r, g, b, 1.0f), "dt:%6.2lfms",
                           DT_ms(m_fps_counter.GetDtAvg()).count());
        ImGui::TextColored(ImVec4(r, g, b, 1.0f), " ±:%6.2lfms",
                           DT_ms(m_fps_counter.GetDtStd()).count());

        const auto offset =
            DT_ms(m_frame_presentation_offset.load(std::memory_order_relaxed)).count();
        ImGui::TextColored(ImVec4(r, g, b, 1.0f), "ofs:%5.1lfms", offset);
      }
    }
    ImGui::End();
  }

  if (g_ActiveConfig.bShowVPS || g_ActiveConfig.bShowVTimes)
  {
    // Position in the top-right corner of the screen.
    ImGui::SetNextWindowPos(ImVec2(window_x, window_y), set_next_position_condition,
                            ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(bg_alpha);

    if (ImGui::Begin("VPSStats", nullptr, imgui_flags))
    {
      if (stack_vertically)
        window_y += ImGui::GetWindowHeight() + window_padding;
      else
        window_x -= ImGui::GetWindowWidth() + window_padding;
      clamp_window_position();
      if (g_ActiveConfig.bShowVPS)
        ImGui::TextColored(ImVec4(r, g, b, 1.0f), "VPS:%7.2lf", vps);
      if (g_ActiveConfig.bShowVTimes)
      {
        ImGui::TextColored(ImVec4(r, g, b, 1.0f), "dt:%6.2lfms",
                           DT_ms(m_vps_counter.GetDtAvg()).count());
        ImGui::TextColored(ImVec4(r, g, b, 1.0f), " ±:%6.2lfms",
                           DT_ms(m_vps_counter.GetDtStd()).count());
      }
    }
    ImGui::End();
  }

  if (g_ActiveConfig.bShowInternalResolution)
  {
    ImGui::SetNextWindowPos(ImVec2(window_x, window_y), set_next_position_condition,
                            ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(bg_alpha);

    if (ImGui::Begin("ResolutionStats", nullptr, imgui_flags))
    {
      if (stack_vertically)
        window_y += ImGui::GetWindowHeight() + window_padding;
      else
        window_x -= ImGui::GetWindowWidth() + window_padding;

      clamp_window_position();

      const FrameBufferSize size = m_frame_buffer_size.load(std::memory_order_relaxed);
      ImGui::TextColored(ImVec4(r, g, b, 1.0f), "XFB res: %ux%u", size.width, size.height);
    }
    ImGui::End();
  }

  ImGui::PopStyleVar(2);
}
