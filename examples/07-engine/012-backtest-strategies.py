import gc

from rich.progress import Progress

from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser
from fintorch.enum import TimeFrame

def main():
    string_args = [
        "--symbols-config", "5",
        "--logger-config", "info"
    ]
    args = DefaultArgumentParser.parse(args=string_args)
    symbols = [args.symbols[4]]
    time_frame = TimeFrame.HOUR4
    engines = [
        engine for engine in args.engines
        if engine.strategy.symbol in symbols
        # and time_frame == engine.strategy.time_frame
    ]
    print(symbols, time_frame, len(engines))
    
    progress = Progress(*RICH_PROGRESS_COLUMNS)
    overall_task_id = progress.add_task(description="overall", total=len(engines))

    with progress:
        for engine in engines:
            try:
                # engine.remove()
                with engine:
                    if 0 < len(engine.clock):
                        task_id = progress.add_task(description=str(engine), total=len(engine.clock))
                        for _ in engine.simulate():
                            progress.update(task_id=task_id, advance=1)

            except Exception as e:
                print(str(engine))
                print(e)
                
            finally:
                progress.update(task_id=task_id, visible=False)
                progress.update(task_id=overall_task_id, advance=1)
                
                del engine
                gc.collect()
                


if __name__ == '__main__':
    main()
