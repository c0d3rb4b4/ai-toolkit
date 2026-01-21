import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';

const execAsync = promisify(exec);

export async function GET() {
  try {
    // Get platform
    const platform = os.platform();
    const isWindows = platform === 'win32';
    const isMac = platform === 'darwin';

    let gpus: any[] = [];
    let hasNvidiaSmi = false;
    let hasMps = false;

    // Check for NVIDIA GPUs
    hasNvidiaSmi = await checkNvidiaSmi(isWindows);
    if (hasNvidiaSmi) {
      const nvidiaGpus = await getGpuStats(isWindows);
      gpus = gpus.concat(nvidiaGpus.map(gpu => ({ ...gpu, type: 'cuda' })));
    }

    // Check for Apple Silicon MPS
    if (isMac) {
      const mpsInfo = await getMpsInfo();
      if (mpsInfo) {
        hasMps = true;
        gpus.push(mpsInfo);
      }
    }

    return NextResponse.json({
      hasNvidiaSmi,
      hasMps,
      platform,
      gpus,
    });
  } catch (error) {
    console.error('Error fetching GPU stats:', error);
    return NextResponse.json(
      {
        hasNvidiaSmi: false,
        hasMps: false,
        platform: os.platform(),
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

async function checkNvidiaSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      // Check if nvidia-smi is available on Windows
      // It's typically located in C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe
      // but we'll just try to run it directly as it may be in PATH
      await execAsync('nvidia-smi -L');
    } else {
      // Linux/macOS check
      await execAsync('which nvidia-smi');
    }
    return true;
  } catch (error) {
    return false;
  }
}

async function getGpuStats(isWindows: boolean) {
  // Command is the same for both platforms, but the path might be different
  const command =
    'nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits';

  // Execute command
  const { stdout } = await execAsync(command, {
    env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
  });

  // Parse CSV output
  const gpus = stdout
    .trim()
    .split('\n')
    .map(line => {
      const [
        index,
        name,
        driverVersion,
        temperature,
        gpuUtil,
        memoryUtil,
        memoryTotal,
        memoryFree,
        memoryUsed,
        powerDraw,
        powerLimit,
        clockGraphics,
        clockMemory,
        fanSpeed,
      ] = line.split(', ').map(item => item.trim());

      return {
        index: parseInt(index),
        name,
        driverVersion,
        temperature: parseInt(temperature),
        utilization: {
          gpu: parseInt(gpuUtil),
          memory: parseInt(memoryUtil),
        },
        memory: {
          total: parseInt(memoryTotal),
          free: parseInt(memoryFree),
          used: parseInt(memoryUsed),
        },
        power: {
          draw: parseFloat(powerDraw),
          limit: parseFloat(powerLimit),
        },
        clocks: {
          graphics: parseInt(clockGraphics),
          memory: parseInt(clockMemory),
        },
        fan: {
          speed: parseInt(fanSpeed) || 0, // Some GPUs might not report fan speed, default to 0
        },
      };
    });

  return gpus;
}

async function getMpsInfo() {
  try {
    // Check if we're on Apple Silicon by checking for Apple Silicon chip
    const { stdout } = await execAsync('sysctl -n machdep.cpu.brand_string');
    const isAppleSilicon = stdout.includes('Apple M');

    if (!isAppleSilicon) {
      return null;
    }

    // Get system info for Apple Silicon
    const { stdout: memInfo } = await execAsync('sysctl -n hw.memsize');
    const totalMemoryBytes = parseInt(memInfo.trim());
    const totalMemoryGB = Math.round(totalMemoryBytes / (1024 * 1024 * 1024));

    // Get current memory usage using vm_stat
    let usedMemoryMB: number | string = 'N/A';
    let freeMemoryMB: number | string = 'N/A';
    
    try {
      const { stdout: vmStat } = await execAsync('vm_stat');
      const pageSize = 4096; // macOS uses 4KB pages
      
      // Parse vm_stat output
      const pagesActive = parseInt(vmStat.match(/Pages active:\s+(\d+)/)?.[1] || '0');
      const pagesWired = parseInt(vmStat.match(/Pages wired down:\s+(\d+)/)?.[1] || '0');
      const pagesInactive = parseInt(vmStat.match(/Pages inactive:\s+(\d+)/)?.[1] || '0');
      const pagesFree = parseInt(vmStat.match(/Pages free:\s+(\d+)/)?.[1] || '0');
      
      // Calculate memory in MB
      const usedPages = pagesActive + pagesWired + pagesInactive;
      usedMemoryMB = Math.round((usedPages * pageSize) / (1024 * 1024));
      freeMemoryMB = Math.round((pagesFree * pageSize) / (1024 * 1024));
    } catch (vmStatError) {
      console.error('Error getting memory stats:', vmStatError);
    }

    // Get GPU cores info
    try {
      const { stdout: gpuCores } = await execAsync('system_profiler SPDisplaysDataType | grep "Total Number of Cores"');
      const gpuCoreCount = parseInt(gpuCores.split(':')[1].trim());

      return {
        index: 0,
        name: 'Apple Silicon GPU',
        type: 'mps' as const,
        model: stdout.trim(),
        cores: gpuCoreCount,
        memory: {
          total: totalMemoryGB * 1024, // Convert GB to MB for consistency with nvidia-smi
          used: usedMemoryMB,
          free: freeMemoryMB,
        },
        utilization: {
          gpu: 'N/A', // macOS doesn't provide GPU utilization easily
          memory: typeof usedMemoryMB === 'number' && typeof totalMemoryGB === 'number'
            ? Math.round((usedMemoryMB / (totalMemoryGB * 1024)) * 100)
            : 'N/A',
        },
        temperature: 'N/A',
        power: {
          draw: 'N/A',
          limit: 'N/A',
        },
        clocks: {
          graphics: 'N/A',
          memory: 'N/A',
        },
        fan: {
          speed: 'N/A',
        },
      };
    } catch (gpuError) {
      // Fallback if we can't get GPU core count
      return {
        index: 0,
        name: 'Apple Silicon GPU',
        type: 'mps' as const,
        model: stdout.trim(),
        memory: {
          total: totalMemoryGB * 1024,
          used: usedMemoryMB,
          free: freeMemoryMB,
        },
        utilization: {
          gpu: 'N/A',
          memory: typeof usedMemoryMB === 'number' && typeof totalMemoryGB === 'number'
            ? Math.round((usedMemoryMB / (totalMemoryGB * 1024)) * 100)
            : 'N/A',
        },
        temperature: 'N/A',
        power: {
          draw: 'N/A',
          limit: 'N/A',
        },
        clocks: {
          graphics: 'N/A',
          memory: 'N/A',
        },
        fan: {
          speed: 'N/A',
        },
      };
    }
  } catch (error) {
    console.error('Error detecting MPS:', error);
    return null;
  }
}
