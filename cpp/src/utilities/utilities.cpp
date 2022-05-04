#include "utilities.h"

void START_CPU_TIMER(SysInfo::Timers &timers)
{
    /* CPU timing */
    timers.CPU = clock();
}

void STOP_CPU_TIMER(SysInfo::Timers &timers)
{
    timers.CPU = clock() - timers.CPU;
    timers.duration = ((double)timers.CPU) / CLOCKS_PER_SEC;
    timers.accum += timers.duration;
}