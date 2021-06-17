"""Definition of GraphicBase class"""
import pathlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

class GraphicBase:
    """Define global parameters of all graphs of the project. Use LaTeX.

    :param suptitle: Suptitle name of the graphic.
    :type suptitle: str

    :param title: Title name of the graphic.
    :type title: str

    :param xlabel: X-axis name.
    :type xlabel: str

    :param ylabel: Y-axis name.
    :type ylabel: str"""
    def __init__(self, suptitle: str, title: str, xlabel: str, ylabel: str,
                 date_format=True):
        # Use LaTeX render
        plt.rcParams.update({'text.usetex' : True,
            'font.family' : 'geometry'})
        # Create the figure
        self.fig, self.ax = plt.subplots(figsize = (30, 15))

        # Mid point of left and right x-positions
        mid = (self.fig.subplotpars.right + self.fig.subplotpars.left)/2

        # Print title and suptitle
        self.fig.suptitle(suptitle, x = mid, weight = 'bold', fontsize = 30)
        self.ax.set_title(title, fontsize = 15, weight = 'bold')

        # Activate the grid
        plt.grid(linestyle = 'dashed', alpha = 0.5)

        # Format the ticks for years
        if date_format:
            years = mdates.YearLocator()   # every year
            years_fmt = mdates.DateFormatter('%Y')

            self.ax.xaxis.set_major_locator(years)
            self.ax.xaxis.set_major_formatter(years_fmt)
            self.ax.xaxis.set_tick_params(rotation = 45, labelsize = 15)
            self.ax.yaxis.set_tick_params(labelsize = 15)
        else:
            self.ax.tick_params(axis='x', labelsize = 27)
            self.ax.tick_params(axis='y', labelsize = 27)

        # Print axis name
        plt.xlabel(xlabel, fontsize = 40, weight = 'bold')
        plt.ylabel(ylabel, fontsize = 40, weight = 'bold')

        # List which contain legend rectangle
        self.__legend_text = {'Plot': [], 'Text': []}


    def add_text_legend(self, text: str, plot = None) -> None:
        """Add `text` into the legend box, the `text` can be associated to a
        `plot`. The method :meth:`show_legend` must be call at the end of all
        :meth:`add_text_legend` calling.

        :param text: Text to add into the legend box.
        :type text: str

        :param plot: Return of `plt.plot()`, to associate `text` to this plot.
        """
        if plot is None :
            # Artist object
            plot = Rectangle((0, 0),
                             1,
                             1,
                             fc = 'w',
                             fill = False,
                             edgecolor = 'none',
                             linewidth = 0)

        self.__legend_text['Plot'].append(plot)
        self.__legend_text['Text'].append(text)

    def show_legend(self, loc='upper left') -> None:
        """Print the legend box."""
        plt.legend(self.__legend_text['Plot'],
            self.__legend_text['Text'],
            loc=loc,
            fontsize=15)

    def show_graph(self) -> None:
        """Show the graphic and close the figure too."""
        plt.show()
        plt.close(self.fig)

    def save_graph(self, result_folder: str, file_name: str) -> None:
        """Save the graphic into the right directory, create the path if it
        doesn't exist. Close the figure too.

        :param result_folder: Path to the directory where the file will be
            saved.
        :type result_folder: str

        :param file_name: Name of the file.
        :type file_name: str"""
        pathlib.Path(result_folder).mkdir(parents = True, exist_ok = True)
        plt.savefig(result_folder + file_name,
                    format = 'pdf',
                    dpi = 1000,
                    bbox_inches = 'tight')
        plt.close(self.fig)
