from utilites.HistoryPlotting import HistoryPlotting


def plot_history(file_path: str = "log/4-log.csv"):
    history_plot = HistoryPlotting(file_path)
    history_plot.plot()
