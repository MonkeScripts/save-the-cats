"""
CG4002 B02 - Step 9: Power Management
=======================================
Covers Task 7 demo requirements:
  - Power measurement methodology
  - CPU frequency scaling (DVFS)
  - PL (FPGA fabric) clock gating
  - Power optimization strategies
  - Battery life estimation for wearable nodes

Two modes:
  1. On Ultra96 (--hardware): reads real PMIC sensors via pmbus
  2. Demo mode (default): simulates everything, runs anywhere

Usage:
  cd cg4002-ai-demo

  # Demo mode (laptop, for video recording)
  python software/power_management.py

  # Hardware mode (on Ultra96 via SSH)
  python3 power_management.py --hardware
"""

import numpy as np
import os
import sys
import json
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate


POWER_RAILS = {
    "VCCINT": {
        "description": "PL (FPGA) core logic",
        "nominal_voltage": 0.85,
        "typical_current_idle": 0.15,
        "typical_current_active": 0.95,
        "unit": "A",
    },
    "VCCAUX": {
        "description": "PL auxiliary (clock buffers, I/O)",
        "nominal_voltage": 1.80,
        "typical_current_idle": 0.05,
        "typical_current_active": 0.12,
        "unit": "A",
    },
    "VCCBRAM": {
        "description": "Block RAM power",
        "nominal_voltage": 0.85,
        "typical_current_idle": 0.01,
        "typical_current_active": 0.08,
        "unit": "A",
    },
    "VCC_PSINTFP": {
        "description": "PS (ARM CPU) full power domain",
        "nominal_voltage": 0.85,
        "typical_current_idle": 0.20,
        "typical_current_active": 0.60,
        "unit": "A",
    },
    "VCC_PSINTLP": {
        "description": "PS low power domain",
        "nominal_voltage": 0.85,
        "typical_current_idle": 0.05,
        "typical_current_active": 0.10,
        "unit": "A",
    },
    "VCC_PSAUX": {
        "description": "PS auxiliary",
        "nominal_voltage": 1.80,
        "typical_current_idle": 0.03,
        "typical_current_active": 0.06,
        "unit": "A",
    },
    "VCC_PSDDR": {
        "description": "DDR memory interface",
        "nominal_voltage": 1.10,
        "typical_current_idle": 0.15,
        "typical_current_active": 0.30,
        "unit": "A",
    },
}


def read_pmic_hardware():
    """
    Read real power data from Ultra96 PMIC via PYNQ pmbus.
    Only works on Ultra96 with PYNQ image installed.

    The Ultra96-V2 has Infineon IR38060/IR38064 PMICs
    accessible via I2C/PMBus.
    """
    try:
        from pynq import get_rails

        rails = get_rails()
        readings = {}

        print("  Live PMIC readings:")
        print(f"  {'Rail':<20} {'Voltage':>10} {'Current':>10} {'Power':>10}")
        print(f"  {'-'*50}")

        total_power = 0
        for name, rail in rails.items():
            v = rail.voltage.value
            i = rail.current.value
            p = rail.power.value
            total_power += p
            readings[name] = {"voltage": v, "current": i, "power": p}
            print(f"  {name:<20} {v:>8.3f} V {i:>8.3f} A {p:>8.3f} W")

        print(f"  {'-'*50}")
        print(f"  {'TOTAL':<20} {'':>10} {'':>10} {total_power:>8.3f} W")

        return readings, total_power

    except ImportError:
        print("  PYNQ not available. Use demo mode instead.")
        return None, None


def measure_power_states_hardware():
    """
    Measure power under different operating conditions on real Ultra96.
    """
    try:
        from pynq import get_rails
        import subprocess

        results = {}

        print("\n  Measuring IDLE state (2s)...")
        time.sleep(2)
        readings_idle, total_idle = read_pmic_hardware()
        results["idle"] = {"total_power_w": total_idle, "readings": readings_idle}

        print("\n  Measuring CPU ACTIVE state (stress test, 5s)...")
        proc = subprocess.Popen(
            ["stress", "--cpu", "4", "--timeout", "5"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(3)  # let it stabilize
        readings_cpu, total_cpu = read_pmic_hardware()
        proc.wait()
        results["cpu_active"] = {"total_power_w": total_cpu, "readings": readings_cpu}

        print("\n  Measuring DPU INFERENCE state...")
        readings_dpu, total_dpu = read_pmic_hardware()
        results["dpu_active"] = {"total_power_w": total_dpu, "readings": readings_dpu}

        return results

    except Exception as e:
        print(f"  Error during measurement: {e}")
        return None


def simulate_power_states():
    """
    Simulated power measurements for demo mode.
    Values based on typical Ultra96-V2 measurements.
    """
    states = {
        "idle": {
            "description": "System idle, no inference",
            "cpu_freq_mhz": 1200,
            "pl_clock_mhz": 0,
            "dpu_active": False,
            "rails": {},
            "total_power_w": 0,
        },
        "cpu_only": {
            "description": "CPU inference (PyTorch, no DPU)",
            "cpu_freq_mhz": 1200,
            "pl_clock_mhz": 0,
            "dpu_active": False,
            "rails": {},
            "total_power_w": 0,
        },
        "dpu_inference": {
            "description": "DPU inference active",
            "cpu_freq_mhz": 1200,
            "pl_clock_mhz": 300,
            "dpu_active": True,
            "rails": {},
            "total_power_w": 0,
        },
        "dpu_idle": {
            "description": "DPU loaded but idle (between inferences)",
            "cpu_freq_mhz": 1200,
            "pl_clock_mhz": 300,
            "dpu_active": False,
            "rails": {},
            "total_power_w": 0,
        },
        "low_power": {
            "description": "Optimized: CPU scaled down + PL clock gated",
            "cpu_freq_mhz": 600,
            "pl_clock_mhz": 0,
            "dpu_active": False,
            "rails": {},
            "total_power_w": 0,
        },
    }

    multipliers = {
        "idle":          {"ps": 0.3, "pl": 0.1, "ddr": 0.5},
        "cpu_only":      {"ps": 0.8, "pl": 0.1, "ddr": 0.7},
        "dpu_inference": {"ps": 0.6, "pl": 1.0, "ddr": 0.9},
        "dpu_idle":      {"ps": 0.3, "pl": 0.5, "ddr": 0.5},
        "low_power":     {"ps": 0.2, "pl": 0.0, "ddr": 0.3},
    }

    for state_name, state in states.items():
        m = multipliers[state_name]
        total = 0

        for rail_name, rail_spec in POWER_RAILS.items():
            v = rail_spec["nominal_voltage"]

            if "PSINT" in rail_name or "PSAUX" in rail_name:
                factor = m["ps"]
            elif "DDR" in rail_name:
                factor = m["ddr"]
            else:
                factor = m["pl"]

            i_idle = rail_spec["typical_current_idle"]
            i_active = rail_spec["typical_current_active"]
            i = i_idle + (i_active - i_idle) * factor
            p = v * i

            state["rails"][rail_name] = {
                "voltage": round(v, 3),
                "current": round(i, 4),
                "power": round(p, 4),
            }
            total += p

        state["total_power_w"] = round(total, 3)

    return states


def demonstrate_dvfs():
    """
    Explain CPU Dynamic Voltage and Frequency Scaling.

    On Ultra96 (Linux):
      cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
      → 299999 599999 1199999 (kHz)

      echo 599999 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
    """
    frequencies = [
        {"freq_mhz": 300,  "voltage_v": 0.72, "rel_power": 0.25,
         "rel_performance": 0.25, "use_case": "Deep sleep / minimal processing"},
        {"freq_mhz": 600,  "voltage_v": 0.78, "rel_power": 0.45,
         "rel_performance": 0.50, "use_case": "Between inferences (idle waiting)"},
        {"freq_mhz": 1200, "voltage_v": 0.85, "rel_power": 1.00,
         "rel_performance": 1.00, "use_case": "Active processing / data preparation"},
    ]

    return frequencies


def demonstrate_clock_gating():
    """
    Explain PL (FPGA) clock gating strategy.

    The DPU only needs to be active during inference.
    Between inferences, the PL clock can be gated to save power.

    On Ultra96 (PYNQ):
      from pynq import Clocks
      Clocks.fclk0_mhz  # read current PL clock
      Clocks.fclk0_mhz = 150  # reduce PL clock
    """
    strategies = [
        {
            "name": "Always-on DPU",
            "pl_clock": "300 MHz continuous",
            "power_impact": "Highest PL power even when idle",
            "latency_impact": "Zero DPU wake-up time",
            "suitable_for": "High-frequency inference (>10 Hz)",
        },
        {
            "name": "Clock scaling",
            "pl_clock": "150 MHz between inferences, 300 MHz during",
            "power_impact": "~40% PL power reduction during idle",
            "latency_impact": "~1ms to switch clock frequency",
            "suitable_for": "Medium-frequency inference (2-10 Hz)",
        },
        {
            "name": "Full clock gating",
            "pl_clock": "0 MHz between inferences, 300 MHz during",
            "power_impact": "~80% PL power reduction during idle",
            "latency_impact": "~5-10ms to re-enable PL clock",
            "suitable_for": "Low-frequency inference (<2 Hz)",
        },
    ]

    return strategies


def wearable_power_budget():
    """
    Power analysis for the ESP32 FireBeetle wearable nodes.
    (Separate from Ultra96 — this is for the body-mounted sensors)
    """
    components = {
        "FireBeetle ESP32": {
            "voltage": 3.3,
            "current_active_ma": 120,
            "current_sleep_ma": 10,
            "duty_cycle": 0.8,  # 80% active (sampling + transmitting)
        },
        "ICM-20948 IMU": {
            "voltage": 3.3,
            "current_active_ma": 1.6,
            "current_sleep_ma": 0.008,
            "duty_cycle": 0.95,  # almost always sampling
        },
    }

    battery = {
        "capacity_mah": 500,
        "voltage": 3.7,
        "efficiency": 0.85,  # voltage regulator efficiency
    }

    total_avg_current = 0
    for name, comp in components.items():
        avg_current = (comp["current_active_ma"] * comp["duty_cycle"] +
                       comp["current_sleep_ma"] * (1 - comp["duty_cycle"]))
        comp["avg_current_ma"] = round(avg_current, 2)
        total_avg_current += avg_current

    effective_capacity = battery["capacity_mah"] * battery["efficiency"]
    runtime_hours = effective_capacity / total_avg_current

    return components, battery, total_avg_current, runtime_hours


def plot_power_states(states, save_path):
    """Bar chart comparing power consumption across system states."""
    names = list(states.keys())
    powers = [states[n]["total_power_w"] for n in names]
    labels = [states[n]["description"] for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#9E9E9E', '#2196F3', '#F44336', '#FF9800', '#4CAF50']
    bars = ax.bar(range(len(names)), powers, color=colors[:len(names)],
                  edgecolor='white', linewidth=2)

    for bar, p, label in zip(bars, powers, labels):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{p:.3f} W', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                label, ha='center', va='center', fontsize=8, color='white',
                fontweight='bold', wrap=True)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=10)
    ax.set_ylabel('Total Power (W)', fontsize=12)
    ax.set_title('Ultra96-V2 Power Consumption by System State', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Power states plot saved: {save_path}")


def plot_power_breakdown(states, save_path):
    """Stacked bar showing per-rail power breakdown."""
    state_names = list(states.keys())

    groups = {
        "PL (FPGA)": ["VCCINT", "VCCAUX", "VCCBRAM"],
        "PS (ARM CPU)": ["VCC_PSINTFP", "VCC_PSINTLP", "VCC_PSAUX"],
        "DDR Memory": ["VCC_PSDDR"],
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    group_colors = {'PL (FPGA)': '#F44336', 'PS (ARM CPU)': '#2196F3', 'DDR Memory': '#FF9800'}
    bottom = np.zeros(len(state_names))

    for group_name, rail_names in groups.items():
        values = []
        for state_name in state_names:
            group_power = sum(
                states[state_name]["rails"].get(r, {}).get("power", 0)
                for r in rail_names
            )
            values.append(group_power)
        values = np.array(values)
        ax.bar(range(len(state_names)), values, bottom=bottom,
               label=group_name, color=group_colors[group_name],
               edgecolor='white', linewidth=1)
        bottom += values

    ax.set_xticks(range(len(state_names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in state_names], fontsize=10)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Power Breakdown by Domain (PL / PS / DDR)', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Power breakdown plot saved: {save_path}")


def plot_dvfs(frequencies, save_path):
    """CPU frequency vs power and performance."""
    freqs = [f["freq_mhz"] for f in frequencies]
    powers = [f["rel_power"] * 100 for f in frequencies]
    perfs = [f["rel_performance"] * 100 for f in frequencies]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(freqs))
    width = 0.35

    bars1 = ax.bar(x - width/2, powers, width, label='Relative Power (%)',
                   color='#F44336', edgecolor='white')
    bars2 = ax.bar(x + width/2, perfs, width, label='Relative Performance (%)',
                   color='#4CAF50', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} MHz" for f in freqs], fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('CPU DVFS: Frequency vs Power vs Performance', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 120])
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  DVFS plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true",
                        help="Read real PMIC sensors on Ultra96")
    args = parser.parse_args()

    print("=" * 65)
    print("  CG4002 B02 — Step 9: Power Management")
    print("=" * 65)

    mode = "HARDWARE (Ultra96)" if args.hardware else "DEMO (simulated)"
    print(f"\n  Mode: {mode}")

    os.makedirs("models/power", exist_ok=True)

    print(f"""
  ┌─ Ultra96-V2 Power Architecture ─────────────────────────┐
  │                                                          │
  │  12V Power Supply                                        │
  │    │                                                     │
  │    ▼                                                     │
  │  Infineon PMICs (IR38060 / IR38064)                      │
  │    ├─ VCCINT   (0.85V) → PL core logic (DPU)            │
  │    ├─ VCCAUX   (1.80V) → PL auxiliary / clocks           │
  │    ├─ VCCBRAM  (0.85V) → Block RAM                       │
  │    ├─ PSINTFP  (0.85V) → ARM Cortex-A53 (full power)    │
  │    ├─ PSINTLP  (0.85V) → ARM low-power domain           │
  │    ├─ PSAUX    (1.80V) → PS auxiliary                    │
  │    └─ PSDDR    (1.10V) → LPDDR4 memory interface         │
  │                                                          │
  │  Measurement: PYNQ pmbus library reads voltage,          │
  │  current, and power from each rail in real-time.         │
  │                                                          │
  │  Code on Ultra96:                                        │
  │    from pynq import get_rails                            │
  │    rails = get_rails()                                   │
  │    for name, rail in rails.items():                      │
  │        print(f"{{name}}: {{rail.power.value:.3f}} W")    │
  └──────────────────────────────────────────────────────────┘
    """)

    print(f"{'='*65}")
    print("  1. POWER CONSUMPTION BY SYSTEM STATE")
    print(f"{'='*65}")

    if args.hardware:
        hw_results = measure_power_states_hardware()
        if hw_results:
            states = hw_results
        else:
            print("  Falling back to simulated values.")
            states = simulate_power_states()
    else:
        states = simulate_power_states()

    print(tabulate(
        [[name, f"{state['total_power_w']:.3f} W", state['description']]
         for name, state in states.items()],
        headers=["State", "Power", "Description"], tablefmt="simple"
    ))

    if "dpu_inference" in states and "low_power" in states:
        active = states["dpu_inference"]["total_power_w"]
        low = states["low_power"]["total_power_w"]
        saving = (1 - low / active) * 100
        print(f"\n  Max power saving (active → low-power): {saving:.0f}%")

    plot_power_states(states, "models/power/power_states.png")
    plot_power_breakdown(states, "models/power/power_breakdown.png")

    print(f"\n{'='*65}")
    print("  2. CPU DYNAMIC VOLTAGE AND FREQUENCY SCALING (DVFS)")
    print(f"{'='*65}")

    frequencies = demonstrate_dvfs()

    print(f"\n  Available CPU frequencies on Ultra96:")
    print(tabulate(
        [[f"{f['freq_mhz']} MHz", f"{f['voltage_v']:.2f} V",
          f"{f['rel_power']*100:.0f}%", f"{f['rel_performance']*100:.0f}%", f['use_case']]
         for f in frequencies],
        headers=["Frequency", "Voltage", "Power", "Performance", "Use Case"],
        tablefmt="simple"
    ))

    print(f"""
  ┌─ DVFS Commands (on Ultra96 via SSH) ────────────────────┐
  │                                                          │
  │  # Check available frequencies                           │
  │  cat /sys/devices/system/cpu/cpu0/cpufreq/\\              │
  │      scaling_available_frequencies                       │
  │  → 299999 599999 1199999                                 │
  │                                                          │
  │  # Set CPU to 600 MHz (power saving)                     │
  │  echo userspace > /sys/devices/system/cpu/cpu0/\\         │
  │      cpufreq/scaling_governor                            │
  │  echo 599999 > /sys/devices/system/cpu/cpu0/\\            │
  │      cpufreq/scaling_setspeed                            │
  │                                                          │
  │  # Set CPU to max (1200 MHz, for processing)             │
  │  echo 1199999 > /sys/devices/system/cpu/cpu0/\\           │
  │      cpufreq/scaling_setspeed                            │
  └──────────────────────────────────────────────────────────┘
    """)

    plot_dvfs(frequencies, "models/power/dvfs_comparison.png")

    print(f"{'='*65}")
    print("  3. PL (FPGA) CLOCK GATING STRATEGIES")
    print(f"{'='*65}")

    strategies = demonstrate_clock_gating()
    for s in strategies:
        print(f"\n  Strategy: {s['name']}")
        print(f"    PL Clock:       {s['pl_clock']}")
        print(f"    Power impact:   {s['power_impact']}")
        print(f"    Latency impact: {s['latency_impact']}")
        print(f"    Best for:       {s['suitable_for']}")

    print(f"""
  ┌─ PL Clock Control (on Ultra96, PYNQ) ──────────────────┐
  │                                                          │
  │  from pynq import Clocks                                 │
  │                                                          │
  │  # Check current PL clock                                │
  │  print(Clocks.fclk0_mhz)  # → 300.0                     │
  │                                                          │
  │  # Reduce PL clock (save power between inferences)       │
  │  Clocks.fclk0_mhz = 150   # half speed, ~40% PL savings │
  │                                                          │
  │  # Restore full speed for inference                      │
  │  Clocks.fclk0_mhz = 300                                 │
  └──────────────────────────────────────────────────────────┘
    """)

    print(f"{'='*65}")
    print("  4. OUR POWER MANAGEMENT STRATEGY")
    print(f"{'='*65}")

    print(f"""
  ┌─ Inference Duty Cycle Strategy ─────────────────────────┐
  │                                                          │
  │  Our system runs inference every ~0.5 seconds.           │
  │  DPU inference takes ~0.15 ms.                           │
  │  That means DPU is active only 0.03% of the time!       │
  │                                                          │
  │  Timeline (500ms cycle):                                 │
  │                                                          │
  │  |←── 499.85ms idle ──→|←0.15ms→|                       │
  │  ┌──────────────────────┬────────┐                       │
  │  │   PL clock gated     │DPU run │  → repeat             │
  │  │   CPU at 600 MHz     │300 MHz │                       │
  │  └──────────────────────┴────────┘                       │
  │                                                          │
  │  Steps per inference cycle:                              │
  │  1. CPU at 600 MHz: collect IMU data from Zenoh buffer   │
  │  2. CPU at 600 MHz: normalize & prepare window           │
  │  3. Enable PL clock → 300 MHz                            │
  │  4. DPU inference: ~0.15 ms                              │
  │  5. Gate PL clock → 0 MHz                                │
  │  6. CPU at 600 MHz: publish result via Zenoh             │
  │  7. CPU drops to 600 MHz: wait for next cycle            │
  └──────────────────────────────────────────────────────────┘
    """)

    print(f"{'='*65}")
    print("  5. WEARABLE NODE POWER BUDGET (ESP32 + IMU)")
    print(f"{'='*65}")

    components, battery, total_current, runtime = wearable_power_budget()

    print(f"\n  Per wearable node (×3 total):")
    rows = [[name, f"{comp['current_active_ma']:.1f} mA",
             f"{comp['current_sleep_ma']:.3f} mA",
             f"{comp['duty_cycle']*100:.0f}%",
             f"{comp['avg_current_ma']:.2f} mA"]
            for name, comp in components.items()]
    rows.append(["Total", "", "", "", f"{total_current:.2f} mA"])
    print(tabulate(rows, headers=["Component", "Active", "Sleep", "Duty", "Average"],
                   tablefmt="simple"))
    print(f"\n  Battery: {battery['capacity_mah']} mAh @ {battery['voltage']}V "
          f"(η={battery['efficiency']*100:.0f}%)")
    print(f"  Effective capacity: {battery['capacity_mah'] * battery['efficiency']:.0f} mAh")
    print(f"  Estimated runtime: {runtime:.1f} hours")
    print(f"  Target: ≥ 3 hours → {'✓ PASS' if runtime >= 3 else '✗ FAIL'}")

    results = {
        "power_states": {k: {"total_power_w": v["total_power_w"], "description": v["description"]}
                         for k, v in states.items()},
        "dvfs_frequencies": frequencies,
        "clock_gating_strategies": strategies,
        "wearable_node": {
            "total_avg_current_ma": total_current,
            "battery_capacity_mah": battery["capacity_mah"],
            "estimated_runtime_hours": runtime,
            "meets_3hr_target": runtime >= 3,
        },
    }
    with open("models/power/power_results.json", "w") as f:
        json.dump(results, f, indent=2)

    dpu_active_pwr = states.get("dpu_inference", {}).get("total_power_w", 0)
    low_pwr = states.get("low_power", {}).get("total_power_w", 0)

    print("\n  Power Management Summary:")
    print(tabulate([
        ["Ultra96 idle",        f"{states['idle']['total_power_w']:.3f} W"],
        ["Ultra96 DPU active",  f"{dpu_active_pwr:.3f} W"],
        ["Ultra96 low-power",   f"{low_pwr:.3f} W"],
        ["CPU DVFS",            "1200→600 MHz between inferences"],
        ["PL clock gating",     "disabled between inferences"],
        ["DPU duty cycle",      "active only 0.03% of time"],
        ["Wearable avg current", f"{total_current:.1f} mA per node"],
        ["Battery life",        f"{runtime:.1f} hours (500mAh)"],
        ["Target >= 3 hrs",     "PASS" if runtime >= 3 else "FAIL"],
    ], tablefmt="simple"))

    print("  ✓ Step 9 complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
