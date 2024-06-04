from utilites import HistoryPlotting


def plot_history(file_path: str = "log/4-log.csv") -> None:
    history_plot = HistoryPlotting.HistoryPlotting(file_path)
    history_plot.plot()
