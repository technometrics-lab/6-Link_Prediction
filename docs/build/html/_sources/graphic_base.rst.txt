Graphic Base
============

GraphicBase Class
-----------------
Definition of the GraphicBase class

*class* **GraphicBase** (suptitle: str, title: str, xlabel: str, ylabel: str)
  Bases: **object**

  Define global parameters of all figures of the project. Uses LaTeX.

  **Parameters** :
    * **suptitle** (str) - suptitle name for the figure.
    * **title** (str) - title name for the figure.
    * **xlabel** (str) - X-axis name.
    * **ylabel** (str) - Y-axis name.
  **add_text_legend** (text: str, plot = None) -> None
    Add text into legend box, the text can be associated to a plot. the method
    **show_legend()** must be called at the end of all **add_text_legend()** calls.

    **Parameters** :
      * **text** (str) - text to add into the legend box.
      * **plot** - return of plt.plot(), to associate the text to.
  **save_graph** (result_folder: str, file_name: str) -> None
    Save the graphic into the right directory, create the path if it doesn't exist.
    Close the figure too.

    **Parameters** :
      * **result_folder** (str) - path to the directory where the file will be saved.
      * **file_name** (str) - name of the file.
  **show_graph** () -> None
    Shows the graphic and closes the figure too.
  **show_legend** () -> None
    Print the legend box.
