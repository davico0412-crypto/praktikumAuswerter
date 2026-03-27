import numpy as np
import pandas as pd
from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
import sys


class PlotterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Plotter Tool")
        self.resize(1200, 800)

        self.max_param_fields = 4
        self.default_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        self.data_columns = ["data1", "data2"]
        self.dataframe = pd.DataFrame(columns=self.data_columns)
        self.column_editors = {"data1": [],
                               "data2": []}
        self.header_name_edits = {}
        self.row_count = 0
        self.x_axis_column = "data1"
        self.y_axis_columns = ["data2"]
        self.x_error_column = ""
        self.y_error_column = ""
        self.show_x_errors = True
        self.show_y_errors = True
        self.use_grouped_averaging = False
        self.group_size = 1
        self.drop_incomplete_groups = False
        self.group_uncertainty_mode = "Std Dev"
        self.plot_mode = "Cartesian"
        self.polar_theta_unit = "Degrees"
        self.dataset_settings = {}
        self.scatter_row_lookup = {}
        self.selected_point = None
        self.guide_lines = []

        self.fit_definitions = {
            "None": {
                "params": [],
                "func": None,
            },
            "Linear: a*x + b": {
                "params": ["a", "b"],
                "func": lambda x, p: p[0] * x + p[1],
            },
            "Quadratic: a*x² + b*x + c": {
                "params": ["a", "b", "c"],
                "func": lambda x, p: p[0] * x**2 + p[1] * x + p[2],
            },
            "Cubic: a*x³ + b*x² + c*x + d": {
                "params": ["a", "b", "c", "d"],
                "func": lambda x, p: p[0] * x**3 + p[1] * x**2 + p[2] * x + p[3],
            },
            "Exponential: a*exp(b*x)": {
                "params": ["a", "b"],
                "func": lambda x, p: p[0] * np.exp(p[1] * x),
            },
            "Sinus: a*sin(b*x +c)+d":{
                "params": ["a", "b","c","d"],
                "func": lambda x, p: p[0] * np.sin((p[1]*x) + p[2]) + p[3],
            },
            "Sinus²: a*sin²(b*x +c)+d":{
                "params": ["a", "b","c","d"],
                "func": lambda x, p: p[0] * np.square(np.sin((p[1]*x) + p[2])) + p[3],
            },
        }

        self._initialize_dataset_settings(self.data_columns)

        self._build_ui()
        self._add_empty_rows(40)
        self._update_axis_combos()
        self._update_plot()

    def _build_ui(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self._build_left_data_panel(splitter)
        self._build_right_panel(splitter)

        splitter.setSizes([420, 780])

    def _build_left_data_panel(self, parent):
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        header = QtWidgets.QLabel("Data Grid")
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        left_layout.addWidget(header)

        top_controls = QtWidgets.QHBoxLayout()
        self.add_dataset_button = QtWidgets.QPushButton("Add Dataset")
        self.add_dataset_button.clicked.connect(self._add_dataset_column)
        top_controls.addWidget(self.add_dataset_button)

        self.import_csv_button = QtWidgets.QPushButton("Import CSV")
        self.import_csv_button.clicked.connect(self._import_csv)
        top_controls.addWidget(self.import_csv_button)

        self.save_data_button = QtWidgets.QPushButton("Save Data")
        self.save_data_button.clicked.connect(self._save_data)
        top_controls.addWidget(self.save_data_button)

        top_controls.addStretch(1)
        left_layout.addLayout(top_controls)

        self.data_scroll_area = QtWidgets.QScrollArea()
        self.data_scroll_area.setWidgetResizable(True)

        self.data_container = QtWidgets.QWidget()
        self.data_grid = QtWidgets.QGridLayout(self.data_container)
        self.data_grid.setContentsMargins(8, 8, 8, 8)
        self.data_grid.setHorizontalSpacing(6)
        self.data_grid.setVerticalSpacing(4)

        self._build_data_grid_header_row()

        self.data_scroll_area.setWidget(self.data_container)
        self.data_scroll_area.verticalScrollBar().valueChanged.connect(self._on_data_scroll)

        left_layout.addWidget(self.data_scroll_area)
        parent.addWidget(left_panel)

    def _build_right_panel(self, parent):
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # Create scroll area for controls
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_container = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_container)

        controls_group = QtWidgets.QGroupBox("Controls")
        controls_layout = QtWidgets.QGridLayout(controls_group)

        controls_layout.addWidget(QtWidgets.QLabel("X Axis:"), 0, 0)
        self.x_axis_combo = QtWidgets.QComboBox()
        self.x_axis_combo.currentTextChanged.connect(self._on_axis_changed)
        controls_layout.addWidget(self.x_axis_combo, 0, 1, 1, 2)

        controls_layout.addWidget(QtWidgets.QLabel("Y Axis (multi):"), 1, 0)
        self.y_axis_scroll = QtWidgets.QScrollArea()
        self.y_axis_scroll.setWidgetResizable(True)
        self.y_axis_container = QtWidgets.QWidget()
        self.y_axis_layout = QtWidgets.QVBoxLayout(self.y_axis_container)
        self.y_axis_layout.setContentsMargins(0, 0, 0, 0)
        self.y_axis_checkboxes = {}
        self.y_axis_scroll.setWidget(self.y_axis_container)
        self.y_axis_scroll.setMaximumHeight(80)
        controls_layout.addWidget(self.y_axis_scroll, 1, 1, 1, 2)

        controls_layout.addWidget(QtWidgets.QLabel("X Error:"), 2, 0)
        self.x_error_combo = QtWidgets.QComboBox()
        self.x_error_combo.currentTextChanged.connect(self._on_error_axis_changed)
        controls_layout.addWidget(self.x_error_combo, 2, 1, 1, 2)

        self.show_x_errors_checkbox = QtWidgets.QCheckBox("Show X Errors")
        self.show_x_errors_checkbox.setChecked(True)
        self.show_x_errors_checkbox.toggled.connect(self._on_error_visibility_changed)
        controls_layout.addWidget(self.show_x_errors_checkbox, 2, 3)

        controls_layout.addWidget(QtWidgets.QLabel("Y Error:"), 3, 0)
        self.y_error_combo = QtWidgets.QComboBox()
        self.y_error_combo.currentTextChanged.connect(self._on_error_axis_changed)
        controls_layout.addWidget(self.y_error_combo, 3, 1, 1, 2)

        self.show_y_errors_checkbox = QtWidgets.QCheckBox("Show Y Errors")
        self.show_y_errors_checkbox.setChecked(True)
        self.show_y_errors_checkbox.toggled.connect(self._on_error_visibility_changed)
        controls_layout.addWidget(self.show_y_errors_checkbox, 3, 3)

        controls_layout.addWidget(QtWidgets.QLabel("Plot Mode:"), 4, 0)
        self.plot_mode_combo = QtWidgets.QComboBox()
        self.plot_mode_combo.addItems(["Cartesian", "Polar"])
        self.plot_mode_combo.currentTextChanged.connect(self._on_plot_mode_changed)
        controls_layout.addWidget(self.plot_mode_combo, 4, 1, 1, 2)

        controls_layout.addWidget(QtWidgets.QLabel("Theta Unit:"), 5, 0)
        self.theta_unit_combo = QtWidgets.QComboBox()
        self.theta_unit_combo.addItems(["Degrees", "Radians"])
        self.theta_unit_combo.currentTextChanged.connect(self._on_plot_mode_changed)
        controls_layout.addWidget(self.theta_unit_combo, 5, 1, 1, 2)

        self.group_average_checkbox = QtWidgets.QCheckBox("Use grouped averaging")
        self.group_average_checkbox.toggled.connect(self._on_grouping_changed)
        controls_layout.addWidget(self.group_average_checkbox, 6, 0, 1, 3)

        controls_layout.addWidget(QtWidgets.QLabel("Group Size:"), 7, 0)
        self.group_size_spinbox = QtWidgets.QSpinBox()
        self.group_size_spinbox.setMinimum(1)
        self.group_size_spinbox.setMaximum(100000)
        self.group_size_spinbox.setValue(1)
        self.group_size_spinbox.valueChanged.connect(self._on_grouping_changed)
        controls_layout.addWidget(self.group_size_spinbox, 7, 1, 1, 2)

        controls_layout.addWidget(QtWidgets.QLabel("Uncertainty:"), 8, 0)
        self.group_uncertainty_combo = QtWidgets.QComboBox()
        self.group_uncertainty_combo.addItems(["Std Dev", "SEM"])
        self.group_uncertainty_combo.currentTextChanged.connect(self._on_grouping_changed)
        controls_layout.addWidget(self.group_uncertainty_combo, 8, 1, 1, 2)

        self.drop_incomplete_checkbox = QtWidgets.QCheckBox("Drop incomplete last group")
        self.drop_incomplete_checkbox.toggled.connect(self._on_grouping_changed)
        controls_layout.addWidget(self.drop_incomplete_checkbox, 9, 0, 1, 3)

        self.point_color_button = QtWidgets.QPushButton("Pick Color")
        self.point_color_button.clicked.connect(self._pick_dataset_color)
        controls_layout.addWidget(self.point_color_button, 10, 0)

        self.clear_button = QtWidgets.QPushButton("Clear Data")
        self.clear_button.clicked.connect(self._clear_data)
        controls_layout.addWidget(self.clear_button, 10, 1, 1, 2)

        controls_layout.addWidget(QtWidgets.QLabel("Plot Title:"), 11, 0)
        self.plot_title_edit = QtWidgets.QLineEdit()
        self.plot_title_edit.setPlaceholderText("title")
        self.plot_title_edit.editingFinished.connect(self._on_plot_labels_changed)
        controls_layout.addWidget(self.plot_title_edit, 11, 1, 1, 2)

        controls_layout.addWidget(QtWidgets.QLabel("Fit Function:"), 12, 0)
        self.fit_combo = QtWidgets.QComboBox()
        self.fit_combo.addItems(self.fit_definitions.keys())
        self.fit_combo.currentTextChanged.connect(self._on_fit_changed)
        controls_layout.addWidget(self.fit_combo, 12, 1, 1, 2)

        controls_layout.addWidget(QtWidgets.QLabel("Fit X Min:"), 13, 0)
        self.fit_x_min_edit = QtWidgets.QLineEdit()
        self.fit_x_min_edit.setPlaceholderText("no min")
        self.fit_x_min_edit.editingFinished.connect(self._on_fit_range_edited)
        controls_layout.addWidget(self.fit_x_min_edit, 13, 1, 1, 2)

        controls_layout.addWidget(QtWidgets.QLabel("Fit X Max:"), 14, 0)
        self.fit_x_max_edit = QtWidgets.QLineEdit()
        self.fit_x_max_edit.setPlaceholderText("no max")
        self.fit_x_max_edit.editingFinished.connect(self._on_fit_range_edited)
        controls_layout.addWidget(self.fit_x_max_edit, 14, 1, 1, 2)

        self.auto_fit_button = QtWidgets.QPushButton("Fit Selected")
        self.auto_fit_button.clicked.connect(self._auto_fit)
        controls_layout.addWidget(self.auto_fit_button, 15, 0)

        self.fit_status = QtWidgets.QLabel("")
        controls_layout.addWidget(self.fit_status, 15, 1, 1, 2)

        controls_layout.addWidget(QtWidgets.QLabel("Picked Point:"), 16, 0)
        self.picked_point_label = QtWidgets.QLabel("none")
        controls_layout.addWidget(self.picked_point_label, 16, 1, 1, 2)

        self.add_vline_button = QtWidgets.QPushButton("Add V-Line")
        self.add_vline_button.clicked.connect(self._add_vline_from_selected_point)
        controls_layout.addWidget(self.add_vline_button, 17, 0)

        self.add_hline_button = QtWidgets.QPushButton("Add H-Line")
        self.add_hline_button.clicked.connect(self._add_hline_from_selected_point)
        controls_layout.addWidget(self.add_hline_button, 17, 1, 1, 2)

        self.clear_lines_button = QtWidgets.QPushButton("Clear Lines")
        self.clear_lines_button.clicked.connect(self._clear_lines)
        controls_layout.addWidget(self.clear_lines_button, 18, 0, 1, 3)

        self.param_labels = []
        self.param_edits = []
        for i in range(self.max_param_fields):
            label = QtWidgets.QLabel(f"p{i}")
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("0.0")
            edit.editingFinished.connect(self._on_param_edited)
            controls_layout.addWidget(label, 19 + i, 0)
            controls_layout.addWidget(edit, 19 + i, 1, 1, 2)
            self.param_labels.append(label)
            self.param_edits.append(edit)

        latex_label_row = 19 + self.max_param_fields
        controls_layout.addWidget(QtWidgets.QLabel("LaTeX Rows:"), latex_label_row, 0)
        self.copy_latex_button = QtWidgets.QPushButton("Copy LaTeX")
        self.copy_latex_button.clicked.connect(self._copy_latex_output)
        controls_layout.addWidget(self.copy_latex_button, latex_label_row, 2)
        self.latex_output_edit = QtWidgets.QPlainTextEdit()
        self.latex_output_edit.setReadOnly(True)
        self.latex_output_edit.setPlaceholderText("1&2&3\\\\\n4&5&6\\\\")
        self.latex_output_edit.setFixedHeight(100)
        controls_layout.addWidget(self.latex_output_edit, latex_label_row + 1, 0, 3, 3)

        self._refresh_line_controls()

        scroll_layout.addWidget(controls_group)
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_container)
        right_layout.addWidget(scroll_area)

        # Add button for image measurement
        button_layout = QtWidgets.QHBoxLayout()
        self.image_measure_button = QtWidgets.QPushButton("Open Image Measurement")
        self.image_measure_button.clicked.connect(self._open_image_measurement)
        button_layout.addWidget(self.image_measure_button)
        right_layout.addLayout(button_layout)

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect("pick_event", self._on_plot_pick)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1)

        parent.addWidget(right_panel)

    def _add_empty_rows(self, count=20):
        start = self.row_count
        for row in range(start, start + count):
            for col_index, col_name in enumerate(self.data_columns):
                edit = QtWidgets.QLineEdit()
                edit.editingFinished.connect(self._on_data_edited)
                self.data_grid.addWidget(edit, row + 1, col_index)
                self.column_editors[col_name].append(edit)

        self.row_count += count
        self._sync_dataframe_from_edits()

    def _on_data_scroll(self, value):
        scrollbar = self.data_scroll_area.verticalScrollBar()
        if value >= scrollbar.maximum() - 6:
            self._add_empty_rows(20)

    def _on_data_edited(self):
        self._sync_dataframe_from_edits()
        self._ensure_trailing_empty_rows()
        self._update_plot()

    def _sync_dataframe_from_edits(self):
        data = {}
        for col_name in self.data_columns:
            data[col_name] = [self._parse_float(edit.text()) for edit in self.column_editors[col_name]]
        self.dataframe = pd.DataFrame(data)
        self._update_latex_output()

    def _update_latex_output(self):
        if not hasattr(self, "latex_output_edit"):
            return

        latex_lines = []
        for row_index in range(self.row_count):
            row_values = [self.column_editors[col_name][row_index].text().strip() for col_name in self.data_columns]
            if all(value == "" for value in row_values):
                continue
            latex_lines.append("&".join(row_values) + r"\\")

        self.latex_output_edit.setPlainText("\n".join(latex_lines))

    def _copy_latex_output(self):
        if not hasattr(self, "latex_output_edit"):
            return

        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(self.latex_output_edit.toPlainText())

    def _ensure_trailing_empty_rows(self):
        trailing_empty = 0
        for row_index in reversed(range(self.row_count)):
            is_empty = True
            for col_name in self.data_columns:
                if self.column_editors[col_name][row_index].text().strip() != "":
                    is_empty = False
                    break
            if is_empty:
                trailing_empty += 1
            else:
                break

        if trailing_empty < 5:
            self._add_empty_rows(10)

    @staticmethod
    def _parse_float(value):
        if value is None:
            return np.nan
        normalized = str(value).strip().replace(",", ".")
        if normalized == "":
            return np.nan
        try:
            return float(normalized)
        except ValueError:
            return np.nan

    def _get_valid_xy(self, x_col, y_col):
        x, y, _ = self._get_valid_xy_with_rows(x_col, y_col)
        return x, y

    def _get_valid_xy_with_rows(self, x_col, y_col):
        if x_col not in self.dataframe.columns or y_col not in self.dataframe.columns:
            return np.array([]), np.array([]), np.array([], dtype=int)

        numeric = self.dataframe[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            return np.array([]), np.array([]), np.array([], dtype=int)
        return valid[x_col].to_numpy(), valid[y_col].to_numpy(), valid.index.to_numpy(dtype=int)

    def _get_valid_xy_errors_with_rows(self, x_col, y_col, xerr_col="", yerr_col=""):
        if x_col not in self.dataframe.columns or y_col not in self.dataframe.columns:
            return np.array([]), np.array([]), None, None, np.array([], dtype=int)

        base_numeric = self.dataframe[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
        valid = base_numeric.dropna()
        if valid.empty:
            return np.array([]), np.array([]), None, None, np.array([], dtype=int)

        x = valid[x_col].to_numpy()
        y = valid[y_col].to_numpy()
        xerr = None
        yerr = None

        valid_indices = valid.index

        if xerr_col and xerr_col in self.dataframe.columns:
            xerr_series = pd.to_numeric(self.dataframe.loc[valid_indices, xerr_col], errors="coerce")
            xerr = np.abs(xerr_series.to_numpy())

        if yerr_col and yerr_col in self.dataframe.columns:
            yerr_series = pd.to_numeric(self.dataframe.loc[valid_indices, yerr_col], errors="coerce")
            yerr = np.abs(yerr_series.to_numpy())

        return x, y, xerr, yerr, valid.index.to_numpy(dtype=int)

    def _pick_dataset_color(self):
        color_col = self.y_axis_columns[0] if self.y_axis_columns else ""
        if color_col == "":
            return

        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.dataset_settings[color_col]["color"] = color.name()
            self._refresh_color_button()
            self._update_plot()

    def _on_axis_changed(self):
        selected_x = self.x_axis_combo.currentData()
        self.x_axis_column = selected_x if selected_x is not None else ""
        
        # Collect checked Y columns
        self.y_axis_columns = [col for col, chk in self.y_axis_checkboxes.items() if chk.isChecked()]
        
        self.fit_status.setText("")
        self._load_selected_dataset_controls()
        self._update_plot()

    def _on_error_visibility_changed(self):
        self.show_x_errors = self.show_x_errors_checkbox.isChecked()
        self.show_y_errors = self.show_y_errors_checkbox.isChecked()
        self._update_plot()

    def _on_error_axis_changed(self):
        selected_x_error = self.x_error_combo.currentData()
        selected_y_error = self.y_error_combo.currentData()
        self.x_error_column = selected_x_error if selected_x_error is not None else ""
        self.y_error_column = selected_y_error if selected_y_error is not None else ""
        self._update_plot()

    def _on_grouping_changed(self):
        self.use_grouped_averaging = self.group_average_checkbox.isChecked()
        self.group_size = max(1, int(self.group_size_spinbox.value()))
        self.drop_incomplete_groups = self.drop_incomplete_checkbox.isChecked()
        self.group_uncertainty_mode = self.group_uncertainty_combo.currentText()
        self._update_plot()

    def _on_plot_mode_changed(self):
        self.plot_mode = self.plot_mode_combo.currentText()
        self.polar_theta_unit = self.theta_unit_combo.currentText()
        self._update_plot()

    def _update_axis_combos(self):
        previous_x = self.x_axis_column if hasattr(self, "x_axis_combo") else self.data_columns[0]
        previous_y = self.y_axis_columns if hasattr(self, "y_axis_checkboxes") else [self.data_columns[1]] if len(self.data_columns) > 1 else [self.data_columns[0]]
        previous_x_error = self.x_error_column if hasattr(self, "x_error_combo") else ""
        previous_y_error = self.y_error_column if hasattr(self, "y_error_combo") else ""
        
        self.x_axis_combo.blockSignals(True)
        
        self.x_axis_combo.clear()
        for col_name in self.data_columns:
            display_name = self._get_column_display_name(col_name)
            self.x_axis_combo.addItem(display_name, col_name)

        x_index = self.x_axis_combo.findData(previous_x)
        if x_index >= 0:
            self.x_axis_combo.setCurrentIndex(x_index)
            self.x_axis_column = previous_x
        elif self.data_columns:
            self.x_axis_combo.setCurrentIndex(0)
            self.x_axis_column = self.data_columns[0]

        self.x_axis_combo.blockSignals(False)

        # Update Y axis checkboxes
        self.y_axis_checkboxes = {}
        # Clear layout
        while self.y_axis_layout.count():
            child = self.y_axis_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        for col_name in self.data_columns:
            if col_name == self.x_axis_column:
                continue
            display_name = self._get_column_display_name(col_name)
            chk = QtWidgets.QCheckBox(display_name)
            chk.toggled.connect(self._on_axis_changed)
            chk.setChecked(col_name in previous_y)
            self.y_axis_layout.addWidget(chk)
            self.y_axis_checkboxes[col_name] = chk
        
        self.y_axis_layout.addStretch()
        self.y_axis_columns = [col for col, chk in self.y_axis_checkboxes.items() if chk.isChecked()]

        self.x_error_combo.blockSignals(True)
        self.y_error_combo.blockSignals(True)

        self.x_error_combo.clear()
        self.y_error_combo.clear()

        self.x_error_combo.addItem("None", "")
        self.y_error_combo.addItem("None", "")
        for col_name in self.data_columns:
            display_name = self._get_column_display_name(col_name)
            self.x_error_combo.addItem(display_name, col_name)
            self.y_error_combo.addItem(display_name, col_name)

        x_error_choice = previous_x_error if previous_x_error in self.data_columns else ""
        y_error_choice = previous_y_error if previous_y_error in self.data_columns else ""

        x_error_index = self.x_error_combo.findData(x_error_choice)
        y_error_index = self.y_error_combo.findData(y_error_choice)
        self.x_error_combo.setCurrentIndex(x_error_index if x_error_index >= 0 else 0)
        self.y_error_combo.setCurrentIndex(y_error_index if y_error_index >= 0 else 0)

        self.x_error_combo.blockSignals(False)
        self.y_error_combo.blockSignals(False)

        self.x_error_column = x_error_choice
        self.y_error_column = y_error_choice

    def _on_fit_changed(self):
        if not self.y_axis_columns:
            return

        dataset_name = self.y_axis_columns[0]
        self.dataset_settings[dataset_name]["fit_name"] = self.fit_combo.currentText()
        self._update_param_fields()
        self._save_params_to_selected_dataset()
        self.fit_status.setText("")
        self._update_plot()

    def _on_fit_range_edited(self):
        self._save_fit_range_to_selected_dataset()
        self.fit_status.setText("")
        self._update_plot()

    def _on_header_name_edited(self, col_name):
        edit = self.header_name_edits.get(col_name)
        if edit is None:
            return

        edited_name = edit.text().strip()
        display_name = edited_name if edited_name else col_name
        self.dataset_settings[col_name]["display_name"] = display_name
        if edited_name == "":
            edit.setText(display_name)
        self._update_axis_combos()
        self._update_plot()

    def _on_plot_labels_changed(self):
        self._update_plot()

    def _update_param_fields(self):
        fit_name = self.fit_combo.currentText()
        param_names = self.fit_definitions[fit_name]["params"]
        dataset_name = self.y_axis_columns[0] if self.y_axis_columns else ""
        params = self.dataset_settings.get(dataset_name, {}).get("params", [np.nan] * self.max_param_fields)

        for i in range(self.max_param_fields):
            is_visible = i < len(param_names)
            self.param_labels[i].setVisible(is_visible)
            self.param_edits[i].setVisible(is_visible)
            if is_visible:
                self.param_labels[i].setText(param_names[i])
                if i < len(params) and not np.isnan(params[i]):
                    self.param_edits[i].setText(f"{params[i]:.6g}")
                elif self.param_edits[i].text().strip() == "":
                    self.param_edits[i].setText("0.0")
            else:
                self.param_edits[i].clear()

    def _on_param_edited(self):
        self._save_params_to_selected_dataset()
        self._update_plot()

    def _save_params_to_selected_dataset(self):
        dataset_name = self.y_axis_columns[0] if self.y_axis_columns else ""
        if dataset_name == "":
            return

        params = self.dataset_settings[dataset_name]["params"]
        for i in range(self.max_param_fields):
            value = self._parse_float(self.param_edits[i].text())
            params[i] = value

    def _save_fit_range_to_selected_dataset(self):
        dataset_name = self.y_axis_columns[0] if self.y_axis_columns else ""
        if dataset_name == "":
            return

        fit_x_min = self._parse_float(self.fit_x_min_edit.text())
        fit_x_max = self._parse_float(self.fit_x_max_edit.text())
        self.dataset_settings[dataset_name]["fit_x_min"] = fit_x_min
        self.dataset_settings[dataset_name]["fit_x_max"] = fit_x_max

    def _get_fit_bounds(self, dataset_name):
        fit_x_min = self.dataset_settings.get(dataset_name, {}).get("fit_x_min", np.nan)
        fit_x_max = self.dataset_settings.get(dataset_name, {}).get("fit_x_max", np.nan)

        if not np.isnan(fit_x_min) and not np.isnan(fit_x_max) and fit_x_min > fit_x_max:
            fit_x_min, fit_x_max = fit_x_max, fit_x_min

        return fit_x_min, fit_x_max

    def _get_fit_mask(self, x_values, fit_x_min, fit_x_max):
        mask = np.ones(len(x_values), dtype=bool)
        if not np.isnan(fit_x_min):
            mask &= x_values >= fit_x_min
        if not np.isnan(fit_x_max):
            mask &= x_values <= fit_x_max
        return mask

    @staticmethod
    def _lighten_color(color_hex, factor=0.55):
        if not isinstance(color_hex, str) or not color_hex.startswith("#") or len(color_hex) != 7:
            return color_hex
        try:
            red = int(color_hex[1:3], 16)
            green = int(color_hex[3:5], 16)
            blue = int(color_hex[5:7], 16)
        except ValueError:
            return color_hex

        red = int(red + (255 - red) * factor)
        green = int(green + (255 - green) * factor)
        blue = int(blue + (255 - blue) * factor)
        return f"#{red:02x}{green:02x}{blue:02x}"

    def _auto_fit(self):
        dataset_name = self.y_axis_columns[0] if self.y_axis_columns else ""
        if dataset_name == "":
            return

        fit_name = self.dataset_settings[dataset_name]["fit_name"]
        if fit_name == "None":
            self.fit_status.setText("select a fit function first")
            return

        fit_def = self.fit_definitions[fit_name]

        x_all, y_all = self._get_valid_xy(self.x_axis_column, dataset_name)
        fit_x_min, fit_x_max = self._get_fit_bounds(dataset_name)
        fit_mask = self._get_fit_mask(x_all, fit_x_min, fit_x_max)
        x = x_all[fit_mask]
        y = y_all[fit_mask]

        if len(x) < max(2, len(fit_def["params"])):
            self.fit_status.setText("not enough points inside fit range")
            return

        try:
            initial_params = []
            for i in range(len(fit_def["params"])):
                value = self._parse_float(self.param_edits[i].text())
                initial_params.append(0.0 if np.isnan(value) else value)

            params, pcov = curve_fit(
                fit_def["func"],
                x,
                y,
                p0=initial_params,
                maxfev=10000,
            )

            uncertainties = np.full(len(params), np.nan)
            if pcov is not None:
                diag = np.diag(pcov)
                valid = diag >= 0
                uncertainties[valid] = np.sqrt(diag[valid])
            
            saved_params = self.dataset_settings[dataset_name]["params"]
            saved_uncertainties = self.dataset_settings[dataset_name]["param_uncertainties"]
            for i, value in enumerate(params):
                saved_params[i] = value
                self.param_edits[i].setText(f"{value:.6g}")
                saved_uncertainties[i] = uncertainties[i]
            self.fit_status.setText(f"fit ok: {dataset_name}")
        except ValueError as exc:
            self.fit_status.setText(str(exc))

        self._update_plot()

    def _read_fit_params(self, dataset_name):
        fit_name = self.dataset_settings[dataset_name]["fit_name"]
        param_names = self.fit_definitions[fit_name]["params"]
        saved_params = self.dataset_settings[dataset_name]["params"]
        params = []

        for i in range(len(param_names)):
            value = saved_params[i]
            if np.isnan(value):
                raise ValueError(f"invalid parameter: {param_names[i]}")
            params.append(value)

        return params

    def _refresh_line_controls(self):
        has_selected_point = self.selected_point is not None
        self.add_vline_button.setEnabled(has_selected_point)
        self.add_hline_button.setEnabled(has_selected_point)

        if not has_selected_point:
            self.picked_point_label.setText("none")
            return

        x_value = self.selected_point["x"]
        y_value = self.selected_point["y"]
        self.picked_point_label.setText(f"x={x_value:.6g}, y={y_value:.6g}")

    def _add_vline_from_selected_point(self):
        if self.selected_point is None:
            return

        self.guide_lines.append({"orientation": "v", "value": self.selected_point["x"]})
        self._update_plot()

    def _add_hline_from_selected_point(self):
        if self.selected_point is None:
            return

        self.guide_lines.append({"orientation": "h", "value": self.selected_point["y"]})
        self._update_plot()

    def _clear_lines(self):
        self.guide_lines = []
        self._update_plot()

    @staticmethod
    def _group_consecutive_measurements(x, y, row_indices, group_size=1, drop_incomplete=False, uncertainty_mode="Std Dev"):
        if group_size <= 1 or len(x) == 0:
            no_uncertainty = np.zeros(len(x), dtype=float)
            return x, y, no_uncertainty, no_uncertainty, row_indices

        grouped_x = []
        grouped_y = []
        grouped_x_unc = []
        grouped_y_unc = []
        grouped_rows = []

        total = len(x)
        for start in range(0, total, group_size):
            end = min(start + group_size, total)
            if drop_incomplete and (end - start) < group_size:
                continue

            x_chunk = x[start:end]
            y_chunk = y[start:end]

            grouped_x.append(float(np.mean(x_chunk)))
            grouped_y.append(float(np.mean(y_chunk)))

            if len(x_chunk) > 1:
                std_x = float(np.std(x_chunk, ddof=1))
                if uncertainty_mode == "SEM":
                    grouped_x_unc.append(std_x / np.sqrt(len(x_chunk)))
                else:
                    grouped_x_unc.append(std_x)
            else:
                grouped_x_unc.append(0.0)

            if len(y_chunk) > 1:
                std_y = float(np.std(y_chunk, ddof=1))
                if uncertainty_mode == "SEM":
                    grouped_y_unc.append(std_y / np.sqrt(len(y_chunk)))
                else:
                    grouped_y_unc.append(std_y)
            else:
                grouped_y_unc.append(0.0)

            grouped_rows.append(int(row_indices[start]))

        if len(grouped_x) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([], dtype=int)

        return (
            np.array(grouped_x, dtype=float),
            np.array(grouped_y, dtype=float),
            np.array(grouped_x_unc, dtype=float),
            np.array(grouped_y_unc, dtype=float),
            np.array(grouped_rows, dtype=int),
        )

    def _update_plot(self):
        is_polar = self.plot_mode == "Polar"
        self.figure.clear()
        if is_polar:
            self.ax = self.figure.add_subplot(111, projection="polar")
        else:
            self.ax = self.figure.add_subplot(111)

        title = self.plot_title_edit.text().strip()
        self.scatter_row_lookup = {}

        x_col = self.x_axis_column
        y_cols = self.y_axis_columns
        
        x_display = self.dataset_settings.get(x_col, {}).get("display_name", x_col)
        
        # Build Y axis label from selected columns
        y_display_names = [self.dataset_settings.get(col, {}).get("display_name", col) for col in y_cols]
        y_label = ", ".join(y_display_names) if y_display_names else "Y Values"
        
        self.ax.set_xlabel(x_display if x_display else x_col)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)

        plotted_any = False
        fit_text_obj = None
        
        if x_col and y_cols:
            for y_idx, y_col in enumerate(y_cols):
                if y_col == x_col:
                    continue
                    
                xerr_col = self.x_error_column
                yerr_col = self.y_error_column
                x, y, xerr, yerr, row_indices = self._get_valid_xy_errors_with_rows(x_col, y_col, xerr_col, yerr_col)

                if self.use_grouped_averaging and len(x) > 0:
                    x, y, grouped_xerr, grouped_yerr, row_indices = self._group_consecutive_measurements(
                        x,
                        y,
                        row_indices,
                        group_size=self.group_size,
                        drop_incomplete=self.drop_incomplete_groups,
                        uncertainty_mode=self.group_uncertainty_mode,
                    )
                    xerr = grouped_xerr
                    yerr = grouped_yerr

                if len(x) > 0:
                    plotted_any = True
                    color = self.dataset_settings[y_col]["color"]
                    excluded_color = self._lighten_color(color)
                    display_name = self.dataset_settings[y_col].get("display_name", y_col)

                    if self.use_grouped_averaging and self.group_size > 1:
                        unc_label = "SEM" if self.group_uncertainty_mode == "SEM" else "std"
                        display_name = f"{display_name} (avg n={self.group_size}, {unc_label})"

                    x_plot = np.deg2rad(x) if is_polar and self.polar_theta_unit == "Degrees" else x

                    fit_x_min, fit_x_max = self._get_fit_bounds(y_col)
                    fit_mask = self._get_fit_mask(x, fit_x_min, fit_x_max)
                    outside_mask = ~fit_mask

                    if np.any(outside_mask):
                        scatter_outside = self.ax.scatter(
                            x_plot[outside_mask],
                            y[outside_mask],
                            color=excluded_color,
                            s=50,
                            label=f"{display_name} (excl. fit)",
                            picker=5,
                        )
                        self.scatter_row_lookup[scatter_outside] = (y_col, row_indices[outside_mask])

                        if (xerr is not None or yerr is not None) and not (y_idx > 0):
                            xerr_outside = xerr[outside_mask] if (xerr is not None and not is_polar and self.show_x_errors) else None
                            yerr_outside = yerr[outside_mask] if (yerr is not None and self.show_y_errors) else None
                            if xerr_outside is not None or yerr_outside is not None:
                                self.ax.errorbar(
                                    x_plot[outside_mask],
                                    y[outside_mask],
                                    xerr=xerr_outside,
                                    yerr=yerr_outside,
                                    fmt="none",
                                    ecolor=excluded_color,
                                    alpha=0.7,
                                    elinewidth=1.2,
                                    capsize=3,
                                )

                    if np.any(fit_mask):
                        scatter_fit = self.ax.scatter(
                            x_plot[fit_mask],
                            y[fit_mask],
                            color=color,
                            s=50,
                            label=f"{display_name} (fit range)",
                            picker=5,
                        )
                        self.scatter_row_lookup[scatter_fit] = (y_col, row_indices[fit_mask])

                        if (xerr is not None or yerr is not None) and not (y_idx > 0):
                            xerr_fit = xerr[fit_mask] if (xerr is not None and not is_polar and self.show_x_errors) else None
                            yerr_fit = yerr[fit_mask] if (yerr is not None and self.show_y_errors) else None
                            if xerr_fit is not None or yerr_fit is not None:
                                self.ax.errorbar(
                                    x_plot[fit_mask],
                                    y[fit_mask],
                                    xerr=xerr_fit,
                                    yerr=yerr_fit,
                                    fmt="none",
                                    ecolor=color,
                                    alpha=0.7,
                                    elinewidth=1.2,
                                    capsize=3,
                                )

                    # Only fit first Y column
                    if y_idx == 0:
                        fit_name = self.dataset_settings[y_col]["fit_name"]
                        fit_def = self.fit_definitions[fit_name]
                        if fit_name != "None" and np.any(fit_mask):
                            try:
                                params = self._read_fit_params(y_col)
                                x_for_fit = x[fit_mask]
                                y_for_fit = y[fit_mask]
                                x_fit = np.linspace(np.min(x_for_fit), np.max(x_for_fit), 500)
                                y_fit = fit_def["func"](x_fit, params)
                                x_fit_plot = np.deg2rad(x_fit) if is_polar and self.polar_theta_unit == "Degrees" else x_fit
                                self.ax.plot(x_fit_plot, y_fit, color=color, linewidth=2.0, linestyle="--", label=f"{display_name} fit")

                                fit_text_lines = [fit_name]
                                if fit_name == "Linear: a*x + b":
                                    param_names = self.fit_definitions[fit_name]["params"]
                                    uncertainties = self.dataset_settings[y_col].get("param_uncertainties", [np.nan] * self.max_param_fields)
                                    param_lines = []
                                    for i, (name, value) in enumerate(zip(param_names, params)):
                                        sigma = uncertainties[i] if i < len(uncertainties) else np.nan
                                        if np.isnan(sigma):
                                            param_lines.append(f"{name} = {value:.3g} ± n/a")
                                        else:
                                            param_lines.append(f"{name} = {value:.3g} ± {sigma:.3g}")
                                else:
                                    param_names = self.fit_definitions[fit_name]["params"]
                                    param_lines = [
                                        f"{name} = {value:.3g}"
                                        for name, value in zip(param_names, params)
                                    ]
                                    y_pred = fit_def["func"](x_for_fit, params)
                                    residuals = y_for_fit - y_pred
                                    ss_res = np.sum(residuals**2)
                                    ss_tot = np.sum((y_for_fit - np.mean(y_for_fit))**2)
                                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                                    fit_text_lines.append(f"R² = {r_squared:.3g}")

                                fit_text = "\n".join(fit_text_lines + param_lines)
                                fit_text_obj = self.ax.text(
                                    0.02,
                                    0.22,
                                    fit_text,
                                    transform=self.ax.transAxes,
                                    va="top",
                                    ha="left",
                                    fontsize=16,
                                    fontweight="bold",
                                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor=color),
                                )
                            except (ValueError, RuntimeError):
                                pass

        if not is_polar:
            for line in self.guide_lines:
                value = line["value"]
                if line["orientation"] == "v":
                    self.ax.axvline(value, linestyle=":", linewidth=2, alpha=0.9)
                    self.ax.text(
                        value,
                        0.98,
                        f"x={value:.6g}",
                        transform=self.ax.get_xaxis_transform(),
                        rotation=90,
                        va="top",
                        ha="right",
                        fontsize=12
                    )
                else:
                    self.ax.axhline(value, linestyle=":", linewidth=2, alpha=0.9)
                    self.ax.text(
                        0.98,
                        value,
                        f"y={value:.6g}",
                        transform=self.ax.get_yaxis_transform(),
                        va="bottom",
                        ha="right",
                        fontsize=12
                    )

        if self.selected_point is not None:
            x_value = self.selected_point["x"]
            y_value = self.selected_point["y"]
            self.ax.scatter(
                [x_value],
                [y_value],
                s=180,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                zorder=6,
            )
            self.ax.text(
                x_value,
                y_value,
                f" ({x_value:.6g}, {y_value:.6g})",
                va="bottom",
                ha="left",
            )

        if plotted_any:
            self.ax.legend()

        self.canvas.draw_idle()

    def _on_plot_pick(self, event):
        scatter = event.artist
        if scatter not in self.scatter_row_lookup:
            return
        if len(event.ind) == 0:
            return

        dataset_name, row_indices = self.scatter_row_lookup[scatter]
        point_index = int(event.ind[0])
        if point_index < 0 or point_index >= len(row_indices):
            return

        offsets = scatter.get_offsets()
        if point_index < len(offsets):
            x_value = float(offsets[point_index][0])
            y_value = float(offsets[point_index][1])
            self.selected_point = {"dataset": dataset_name, "x": x_value, "y": y_value}
            self._refresh_line_controls()
        else:
            self._refresh_line_controls()
            return
        self._update_plot()
        row_index = int(row_indices[point_index])
        self._focus_data_cell(dataset_name, row_index)

    def _focus_data_cell(self, col_name, row_index):
        if col_name not in self.column_editors:
            return
        if row_index < 0 or row_index >= len(self.column_editors[col_name]):
            return

        target_edit = self.column_editors[col_name][row_index]
        self.data_scroll_area.ensureWidgetVisible(target_edit, 20, 20)
        target_edit.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        target_edit.selectAll()

    def _clear_data(self):
        dataset_name = self.y_axis_columns[0] if self.y_axis_columns else ""
        if dataset_name == "":
            return

        for edit in self.column_editors[dataset_name]:
            edit.clear()

        if self.selected_point is not None and self.selected_point["dataset"] == dataset_name:
            self.selected_point = None
            self._refresh_line_controls()

        self.dataset_settings[dataset_name]["fit_name"] = "None"
        self.dataset_settings[dataset_name]["params"] = [np.nan] * self.max_param_fields
        self.dataset_settings[dataset_name]["param_uncertainties"] = [np.nan] * self.max_param_fields
        self.dataset_settings[dataset_name]["fit_x_min"] = np.nan
        self.dataset_settings[dataset_name]["fit_x_max"] = np.nan

        self.fit_status.setText("")
        self._load_selected_dataset_controls()
        self._sync_dataframe_from_edits()
        self._update_plot()

    def _initialize_dataset_settings(self, dataset_names):
        current_names = set(self.dataset_settings.keys())
        target_names = set(dataset_names)

        for name in current_names - target_names:
            del self.dataset_settings[name]

        for index, name in enumerate(dataset_names):
            if name not in self.dataset_settings:
                self.dataset_settings[name] = {
                    "color": self.default_colors[index % len(self.default_colors)],
                    "display_name": name,
                    "fit_name": "None",
                    "params": [np.nan] * self.max_param_fields,
                    "param_uncertainties": [np.nan] * self.max_param_fields,
                    "fit_x_min": np.nan,
                    "fit_x_max": np.nan,
                }
            else:
                self.dataset_settings[name].setdefault("param_uncertainties", [np.nan] * self.max_param_fields)
                self.dataset_settings[name].setdefault("fit_x_min", np.nan)
                self.dataset_settings[name].setdefault("fit_x_max", np.nan)

    def _refresh_color_button(self):
        dataset_name = self.y_axis_columns[0] if self.y_axis_columns else ""
        if dataset_name == "":
            self.point_color_button.setStyleSheet("")
            return
        color = self.dataset_settings[dataset_name]["color"]
        self.point_color_button.setStyleSheet(f"background-color: {color};")

    def _load_selected_dataset_controls(self):
        # Use first selected Y column for controls
        dataset_name = self.y_axis_columns[0] if self.y_axis_columns else ""
        controls_enabled = dataset_name != ""

        self.point_color_button.setEnabled(controls_enabled)
        self.fit_combo.setEnabled(controls_enabled)
        self.auto_fit_button.setEnabled(controls_enabled)
        self.fit_x_min_edit.setEnabled(controls_enabled)
        self.fit_x_max_edit.setEnabled(controls_enabled)

        if not controls_enabled:
            self.fit_x_min_edit.clear()
            self.fit_x_max_edit.clear()
            self._refresh_color_button()
            return

        setting = self.dataset_settings[dataset_name]
        self.fit_combo.blockSignals(True)
        self.fit_combo.setCurrentText(setting["fit_name"])
        self.fit_combo.blockSignals(False)

        fit_x_min = setting.get("fit_x_min", np.nan)
        fit_x_max = setting.get("fit_x_max", np.nan)
        self.fit_x_min_edit.setText("" if np.isnan(fit_x_min) else f" {fit_x_min:.6g}")
        self.fit_x_max_edit.setText("" if np.isnan(fit_x_max) else f"{fit_x_max:.6g}")

        self._refresh_color_button()
        self._update_param_fields()

    def _add_dataset_column(self):
        index = len(self.data_columns) + 1
        new_name = f"data{index}"
        while new_name in self.data_columns:
            index += 1
            new_name = f"data{index}"
        self.data_columns.append(new_name)
        self.dataframe[new_name] = np.nan

        self.column_editors[new_name] = []
        col_index = len(self.data_columns) - 1

        label = QtWidgets.QLineEdit()
        label.setText(self._get_column_display_name(new_name))
        label.setStyleSheet("font-weight: bold;")
        label.editingFinished.connect(lambda col_name=new_name: self._on_header_name_edited(col_name))
        self.data_grid.addWidget(label, 0, col_index)
        self.header_name_edits[new_name] = label

        for row_index in range(self.row_count):
            edit = QtWidgets.QLineEdit()
            edit.editingFinished.connect(self._on_data_edited)
            self.data_grid.addWidget(edit, row_index + 1, col_index)
            self.column_editors[new_name].append(edit)

        self._initialize_dataset_settings(self.data_columns)
        self._update_axis_combos()
        self._load_selected_dataset_controls()
        self._sync_dataframe_from_edits()
        self._update_plot()

    def _import_csv(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import TXT",
            "",
            "Text files (*.txt);; All files (*)",
        )

        if file_path == "":
            return

        try:
            imported = pd.read_csv(file_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "TXT import", f"Could not read TXT:\n{exc}")
            return

        if imported.shape[1] == 0:
            QtWidgets.QMessageBox.warning(self, "TXT import", "TXT has no columns.")
            return

        columns = []
        for i, original_name in enumerate(imported.columns):
            cleaned = str(original_name).strip()
            if cleaned == "":
                cleaned = f"data{i+1}"
            while cleaned in columns:
                cleaned = f"{cleaned}_1"
            columns.append(cleaned)

        if len(columns) == 1:
            imported["data1"] = np.nan
            columns.append("data1")

        imported.columns = columns
        self.data_columns = columns
        self.dataframe = imported.copy()
        self._rebuild_grid_from_dataframe(extra_rows=20)
        self._initialize_dataset_settings(self.data_columns)
        self._update_axis_combos()
        self._load_selected_dataset_controls()
        self._update_plot()

    def _save_data(self):
        self._sync_dataframe_from_edits()

        export_df = self.dataframe.dropna(how="all").reset_index(drop=True)
        rename_map = {col_name: self._get_column_display_name(col_name) for col_name in self.data_columns}
        export_df = export_df.rename(columns=rename_map)

        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Data",
            "",
            "Text files (*.txt);;All files (*)",
        )

        if file_path == "":
            return

        selected_filter = selected_filter.lower()
        is_txt = "*.txt" in selected_filter
        is_csv = "*.csv" in selected_filter

        if file_path.lower().endswith(".txt"):
            is_txt = True
            is_csv = False
        elif file_path.lower().endswith(".csv"):
            is_csv = True
            is_txt = False
        elif is_txt:
            file_path = f"{file_path}.txt"
        else:
            file_path = f"{file_path}.csv"
            is_csv = True

        try:
            if is_txt:
                export_df.to_csv(file_path, sep="\t", index=False, na_rep="")
            else:
                export_df.to_csv(file_path, sep=",", index=False, na_rep="")
            QtWidgets.QMessageBox.information(self, "Save Data", f"Saved:\n{file_path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save Data", f"Could not save file:\n{exc}")

    def _rebuild_grid_from_dataframe(self, extra_rows=20):
        source_df = self.dataframe.copy(deep=True)

        old_widget = self.data_scroll_area.takeWidget()
        if old_widget is not None:
            old_widget.deleteLater()

        self.data_container = QtWidgets.QWidget()
        self.data_grid = QtWidgets.QGridLayout(self.data_container)
        self.data_grid.setContentsMargins(8, 8, 8, 8)
        self.data_grid.setHorizontalSpacing(6)
        self.data_grid.setVerticalSpacing(4)

        self.column_editors = {col_name: [] for col_name in self.data_columns}
        self.header_name_edits = {}

        self._build_data_grid_header_row()

        row_total = len(source_df.index) + extra_rows
        self.row_count = 0
        self.data_scroll_area.setWidget(self.data_container)
        self._add_empty_rows(row_total)

        for row_index in range(len(source_df)):
            for col_name in self.data_columns:
                value = source_df.at[row_index, col_name]
                if pd.notna(value):
                    self.column_editors[col_name][row_index].setText(str(value))

        self._sync_dataframe_from_edits()

    def _build_data_grid_header_row(self):
        self.header_name_edits = {}
        for col_index, col_name in enumerate(self.data_columns):
            header_edit = QtWidgets.QLineEdit()
            header_edit.setText(self._get_column_display_name(col_name))
            header_edit.setStyleSheet("font-weight: bold;")
            header_edit.editingFinished.connect(lambda col_name=col_name: self._on_header_name_edited(col_name))
            self.data_grid.addWidget(header_edit, 0, col_index)
            self.header_name_edits[col_name] = header_edit

    def _get_column_display_name(self, col_name):
        if col_name in self.dataset_settings:
            return self.dataset_settings[col_name].get("display_name", col_name)
        return col_name

    def _open_image_measurement(self):
        self.image_window = ImageMeasurementWindow(parent=self)
        self.image_window.show()
    
    def _import_measurement_data(self, column_name, data):
        """Import measurement data from ImageMeasurementWindow as a new column."""
        if column_name not in self.data_columns:
            self.data_columns.append(column_name)
            self.column_editors[column_name] = []
            self.dataset_settings[column_name] = {
                "display_name": column_name,
                "color": self.default_colors[len(self.data_columns) - 1 % len(self.default_colors)],
                "fit_name": "None",
                "params": [np.nan] * self.max_param_fields,
                "fit_x_min": np.nan,
                "fit_x_max": np.nan,
            }
            self._build_data_grid_header_row()
        
        # Get column index
        col_index = self.data_columns.index(column_name)
        
        # Ensure all columns have the same number of rows as the data 
        max_index = max([len(self.column_editors[col]) for col in self.data_columns if col != column_name] + [0])
        target_rows = max(len(data), max_index)
        
        # Extend all columns to target_rows if needed
        for col_name in self.data_columns:
            while len(self.column_editors[col_name]) < target_rows:
                row_index = len(self.column_editors[col_name])
                edit = QtWidgets.QLineEdit()
                edit.editingFinished.connect(self._on_data_edited)
                col_idx = self.data_columns.index(col_name)
                self.data_grid.addWidget(edit, row_index + 1, col_idx)
                self.column_editors[col_name].append(edit)
        
        # Fill in the measurement data
        for i, value in enumerate(data):
            self.column_editors[column_name][i].setText(str(value))
        
        self._sync_dataframe_from_edits()
        self._update_axis_combos()
        self._update_plot()


class ImageMeasurementWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Measurement Tool")
        self.resize(1000, 700)
        
        self.image_path = None
        self.image = None
        self.pixmap = None
        self.drawing = False
        self.points = []
        self.measurements = []
        self.scale_mm_per_pixel = 1.0
        
        self._build_ui()
    
    def _build_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        
        # Top controls
        top_layout = QtWidgets.QHBoxLayout()
        
        load_button = QtWidgets.QPushButton("Load Image")
        load_button.clicked.connect(self._load_image)
        top_layout.addWidget(load_button)
        
        top_layout.addWidget(QtWidgets.QLabel("Scale (mm/pixel):"))
        self.scale_edit = QtWidgets.QLineEdit()
        self.scale_edit.setText("1.0")
        self.scale_edit.setMaximumWidth(100)
        self.scale_edit.editingFinished.connect(self._on_scale_changed)
        top_layout.addWidget(self.scale_edit)
        
        top_layout.addWidget(QtWidgets.QLabel("Measurement Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Length", "Area", "Clear Points"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        top_layout.addWidget(self.mode_combo)
        
        restore_button = QtWidgets.QPushButton("Restore Image")
        restore_button.clicked.connect(self._restore_image)
        top_layout.addWidget(restore_button)
        
        export_button = QtWidgets.QPushButton("Export Measurements")
        export_button.clicked.connect(self._export_measurements)
        top_layout.addWidget(export_button)
        
        top_layout.addStretch()
        main_layout.addLayout(top_layout)
        
        # Image display area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.image_label.mousePressEvent = self._on_image_click
        scroll_area.setWidget(self.image_label)
        main_layout.addWidget(scroll_area, stretch=1)
        
        # Results display
        results_label = QtWidgets.QLabel("Measurements:")
        font = results_label.font()
        font.setBold(True)
        results_label.setFont(font)
        main_layout.addWidget(results_label)
        
        self.results_text = QtWidgets.QPlainTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(120)
        main_layout.addWidget(self.results_text)
    
    def _load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        if file_path:
            self.image_path = file_path
            self.image = QtGui.QImage(file_path)
            self.original_image = QtGui.QImage(file_path)
            self.points = []
            self.measurements = []
            self._display_image()
            self.results_text.setPlainText("Image loaded. Click on the image to set points.")
    
    def _display_image(self):
        if self.image is None:
            return
        
        self.pixmap = QtGui.QPixmap.fromImage(self.image)
        scaled_pixmap = self.pixmap.scaledToWidth(800, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
    
    def _restore_image(self):
        if self.original_image is None:
            return
        
        self.image = QtGui.QImage(self.original_image)
        self.points = []
        self.measurements = []
        self._display_image()
        self.results_text.setPlainText("Image restored. Points cleared.")
    
    def _on_scale_changed(self):
        try:
            self.scale_mm_per_pixel = float(self.scale_edit.text())
        except ValueError:
            self.scale_mm_per_pixel = 1.0
            self.scale_edit.setText("1.0")
    
    def _on_mode_changed(self):
        mode = self.mode_combo.currentText()
        if mode == "Clear Points":
            self.points = []
            self.measurements = []
            self._display_image()
            self.results_text.setPlainText("Points cleared.")
            self.mode_combo.setCurrentIndex(0)
    
    def _on_image_click(self, event):
        if self.image is None:
            return
        
        # Get click position relative to the scaled image
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap = self.image_label.pixmap()
        
        if pixmap is None:
            return
        
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        
        # Calculate offset (centered image)
        x_offset = (label_width - pixmap_width) / 2
        y_offset = (label_height - pixmap_height) / 2
        
        # Get click position in image coordinates
        x_scaled = event.pos().x() - x_offset
        y_scaled = event.pos().y() - y_offset
        
        # Scale back to original image coordinates
        scale_factor = self.image.width() / pixmap_width if pixmap_width > 0 else 1
        x = int(x_scaled * scale_factor)
        y = int(y_scaled * scale_factor)
        
        # Clamp to image bounds
        x = max(0, min(x, self.image.width() - 1))
        y = max(0, min(y, self.image.height() - 1))
        
        self.points.append((x, y))
        self._update_measurements()
        self._draw_points()
    
    def _update_measurements(self):
        mode = self.mode_combo.currentText()
        self.measurements = []
        
        if mode == "Length" and len(self.points) >= 2:
            # Calculate lengths between consecutive points
            for i in range(len(self.points) - 1):
                x1, y1 = self.points[i]
                x2, y2 = self.points[i + 1]
                pixel_dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                mm_dist = pixel_dist * self.scale_mm_per_pixel
                self.measurements.append(("Length", mm_dist, pixel_dist))
        
        elif mode == "Area" and len(self.points) >= 3:
            # Calculate area using Shoelace formula
            area_pixels = self._calculate_polygon_area(self.points)
            area_mm = area_pixels * (self.scale_mm_per_pixel ** 2)
            self.measurements.append(("Area", area_mm, area_pixels))
        
        self._display_results()
    
    def _calculate_polygon_area(self, points):
        if len(points) < 3:
            return 0
        
        area = 0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        
        return abs(area) / 2
    
    def _draw_points(self):
        if self.image is None:
            return
        
        # Restore original image
        self.image = QtGui.QImage(self.original_image)
        
        painter = QtGui.QPainter(self.image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Draw points
        for i, (x, y) in enumerate(self.points):
            painter.fillRect(x - 3, y - 3, 6, 6, QtCore.Qt.GlobalColor.red)
            painter.drawText(x + 5, y - 5, f"{i}")
        
        # Draw lines for length mode
        mode = self.mode_combo.currentText()
        if mode == "Length" and len(self.points) > 1:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.blue, 2)
            painter.setPen(pen)
            for i in range(len(self.points) - 1):
                x1, y1 = self.points[i]
                x2, y2 = self.points[i + 1]
                painter.drawLine(x1, y1, x2, y2)
        
        # Draw polygon for area mode
        elif mode == "Area" and len(self.points) > 1:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.green, 2)
            painter.setPen(pen)
            for i in range(len(self.points)):
                x1, y1 = self.points[i]
                x2, y2 = self.points[(i + 1) % len(self.points)]
                painter.drawLine(x1, y1, x2, y2)
        
        painter.end()
        self._display_image()
    
    def _display_results(self):
        if not self.measurements:
            self.results_text.setPlainText("Click points on the image to measure.")
            return
        
        results = []
        for meas_type, value_mm, value_pixels in self.measurements:
            if meas_type == "Length":
                results.append(f"Length: {value_mm:.2f} mm ({value_pixels:.2f} pixels)")
            elif meas_type == "Area":
                results.append(f"Area: {value_mm:.2f} mm² ({value_pixels:.2f} pixels²)")
        
        # Total calculations
        if self.mode_combo.currentText() == "Length":
            total_mm = sum(m[1] for m in self.measurements if m[0] == "Length")
            results.append(f"\nTotal Length: {total_mm:.2f} mm")
        elif self.mode_combo.currentText() == "Area":
            total_area = sum(m[1] for m in self.measurements if m[0] == "Area")
            results.append(f"\nTotal Area: {total_area:.2f} mm²")
        
        self.results_text.setPlainText("\n".join(results))
    
    def _export_measurements(self):
        if not self.measurements:
            QtWidgets.QMessageBox.warning(self, "No Measurements", "No measurements available to export.")
            return
        
        # Create measurement data
        data = []
        mode = self.mode_combo.currentText()
        
        if mode == "Length":
            for i, (meas_type, value_mm, value_pixels) in enumerate(self.measurements):
                data.append(value_mm)
        elif mode == "Area":
            for i, (meas_type, value_mm, value_pixels) in enumerate(self.measurements):
                data.append(value_mm)
        
        # Dialog to get column name
        dialog = QtWidgets.QInputDialog()
        col_name, ok = dialog.getText(
            self, "Export Measurements",
            "Enter name for new data column:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"measurement_{mode.lower()}"
        )
        
        if not ok or not col_name.strip():
            return
        
        col_name = col_name.strip()
        
        # Use parent if set
        if self.parent() is not None and isinstance(self.parent(), PlotterWindow):
            self.parent()._import_measurement_data(col_name, data)
        else:
            # Fallback: Try to find PlotterWindow in the application
            app = QtWidgets.QApplication.instance()
            for widget in app.topLevelWidgets():
                if isinstance(widget, PlotterWindow):
                    widget._import_measurement_data(col_name, data)
                    break
        
        QtWidgets.QMessageBox.information(
            self, "Export Successful",
            f"Measurements exported to column '{col_name}' in the Plotter."
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PlotterWindow()
    window.show()
    sys.exit(app.exec())
