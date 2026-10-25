"""Run one part of the demo as `python -m demo.run_part MODULE ARGS...`, so the launcher can stop it cleanly on Windows.

The launcher starts each part in its own process group and stops it with CTRL_BREAK (Ctrl-C can't be sent to one
group on Windows). Python's default for CTRL_BREAK is to exit at once, skipping every `except KeyboardInterrupt` and
`finally`: the logger's last records, the websockets, the serial port. Here CTRL_BREAK raises KeyboardInterrupt,
exactly like Ctrl-C, before the module runs; elsewhere this changes nothing.
"""
import runpy
import signal
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: python -m demo.run_part MODULE [ARGS...]')
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
