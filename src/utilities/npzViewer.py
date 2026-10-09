#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ Viewer con soporte para especies
====================================

Visor para archivos .npz que incluye:
- Inspección de variables, dimensiones, estadísticas y datos tabulados.
- Gráficos XY.
- Contornos 2D generales.
- Representación específica de especies almacenadas como Y(Nz, Ns, Nc).
- Selección independiente de coordenadas X, Y y especie.
- Perfiles para un índice Nc, comparación de especies y mapas 2D.

Instalación:
    python -m pip install numpy matplotlib PyQt5

Ejecución:
    python npz_viewer_species_xy.py
"""

import os
import sys
from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


MAX_TABLE_ROWS = 1000
MAX_TABLE_COLUMNS = 500


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 5), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def new_axes(self):
        self.figure.clear()
        return self.figure.add_subplot(111)


class NPZViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.npz = None
        self.current_file = None

        self.setWindowTitle("NPZ Viewer")
        self.resize(1500, 920)

        self._create_actions()
        self._create_menu()
        self._create_ui()
        self.statusBar().showMessage("Abra un archivo NPZ para comenzar")

    # ------------------------------------------------------------------
    # Interfaz principal
    # ------------------------------------------------------------------
    def _create_actions(self):
        self.open_action = QAction("Abrir NPZ...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.triggered.connect(self.open_npz)

        self.export_action = QAction("Exportar variable a CSV...", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self.export_csv)
        self.export_action.setEnabled(False)

        self.exit_action = QAction("Salir", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)

    def _create_menu(self):
        menu = self.menuBar().addMenu("Archivo")
        menu.addAction(self.open_action)
        menu.addAction(self.export_action)
        menu.addSeparator()
        menu.addAction(self.exit_action)

    def _create_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        open_button = QPushButton("Abrir NPZ")
        open_button.clicked.connect(self.open_npz)
        self.file_label = QLabel("Ningún archivo abierto")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(open_button)
        top.addWidget(self.file_label, 1)
        root.addLayout(top)

        main_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(main_splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Variables del archivo"))
        self.variable_list = QListWidget()
        self.variable_list.currentTextChanged.connect(self.on_variable_selected)
        left_layout.addWidget(self.variable_list)
        main_splitter.addWidget(left)

        right_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(right_splitter)

        self.data_tabs = QTabWidget()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setLineWrapMode(QTextEdit.NoWrap)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.data_tabs.addTab(self.info_text, "Información")
        self.data_tabs.addTab(self.table, "Datos")
        right_splitter.addWidget(self.data_tabs)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(self._build_quick_tab(), "Vista rápida")
        self.plot_tabs.addTab(self._build_xy_tab(), "Gráfico XY")
        self.plot_tabs.addTab(self._build_contour_tab(), "Contorno 2D")
        self.plot_tabs.addTab(self._build_species_tab(), "Especies")
        plot_layout.addWidget(self.plot_tabs)

        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        right_splitter.addWidget(plot_panel)

        main_splitter.setSizes([270, 1230])
        right_splitter.setSizes([300, 620])

    def _build_quick_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        self.quick_label = QLabel("Seleccione una variable en la lista.")
        button = QPushButton("Representar variable seleccionada")
        button.clicked.connect(self.plot_quick_view)
        layout.addWidget(self.quick_label, 1)
        layout.addWidget(button)
        return page

    def _build_xy_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)

        self.xy_x_combo = QComboBox()
        self.xy_y_combo = QComboBox()
        self.xy_style_combo = QComboBox()
        self.xy_style_combo.addItems(
            ["Línea", "Línea y marcadores", "Marcadores"]
        )

        form = QFormLayout()
        form.addRow("Variable X:", self.xy_x_combo)
        form.addRow("Variable Y:", self.xy_y_combo)
        form.addRow("Estilo:", self.xy_style_combo)

        button = QPushButton("Dibujar XY")
        button.clicked.connect(self.plot_xy)
        layout.addLayout(form, 1)
        layout.addWidget(button)
        return page

    def _build_contour_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)

        self.contour_x_combo = QComboBox()
        self.contour_y_combo = QComboBox()
        self.contour_z_combo = QComboBox()
        self.contour_type_combo = QComboBox()
        self.contour_type_combo.addItems(
            ["Contorno relleno", "Líneas", "Pseudocolor"]
        )
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            ["viridis", "turbo", "jet", "plasma", "inferno", "magma", "coolwarm", "cividis"]
        )
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(2, 200)
        self.levels_spin.setValue(40)

        form = QFormLayout()
        form.addRow("Coordenada X:", self.contour_x_combo)
        form.addRow("Coordenada Y:", self.contour_y_combo)
        form.addRow("Variable representada:", self.contour_z_combo)
        form.addRow("Tipo:", self.contour_type_combo)
        form.addRow("Mapa de colores:", self.colormap_combo)
        form.addRow("Niveles:", self.levels_spin)

        button = QPushButton("Dibujar contorno")
        button.clicked.connect(self.plot_contour)
        layout.addLayout(form, 1)
        layout.addWidget(button)
        return page

    def _build_species_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)

        self.species_data_combo = QComboBox()
        self.species_x_combo = QComboBox()
        self.species_ycoord_combo = QComboBox()
        self.species_combo = QComboBox()

        self.species_nc_spin = QSpinBox()
        self.species_nc_spin.setRange(0, 0)
        self.species_nc_spin.setToolTip(
            "Selecciona una columna Nc para representar un perfil"
        )

        self.species_all_nc_check = QCheckBox(
            "Representar todos los perfiles Nc"
        )
        self.species_all_nc_check.toggled.connect(
            lambda checked: self.species_nc_spin.setEnabled(not checked)
        )

        self.species_plot_type_combo = QComboBox()
        self.species_plot_type_combo.addItems(
            ["Contorno relleno", "Líneas", "Pseudocolor"]
        )

        self.species_coord_mode_label = QLabel(
            "Para el mapa 2D, X e Y deben describir la malla de Y[:, k, :]."
        )
        self.species_coord_mode_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Array de especies Y(Nz, Ns, Nc):", self.species_data_combo)
        form.addRow("Coordenada X:", self.species_x_combo)
        form.addRow("Coordenada Y:", self.species_ycoord_combo)
        form.addRow("Especie:", self.species_combo)
        form.addRow("Índice Nc para perfiles:", self.species_nc_spin)
        form.addRow("", self.species_all_nc_check)
        form.addRow("Tipo de mapa 2D:", self.species_plot_type_combo)
        form.addRow("", self.species_coord_mode_label)

        buttons = QVBoxLayout()
        profile_x_button = QPushButton("Perfil: X frente a especie")
        profile_x_button.clicked.connect(
            lambda: self.plot_species_profile("x")
        )
        profile_y_button = QPushButton("Perfil: Y frente a especie")
        profile_y_button.clicked.connect(
            lambda: self.plot_species_profile("y")
        )
        map_button = QPushButton("Mapa 2D X-Y de la especie")
        map_button.clicked.connect(self.plot_species_xy_map)
        compare_button = QPushButton("Comparar todas las especies")
        compare_button.clicked.connect(self.plot_all_species)

        buttons.addWidget(profile_x_button)
        buttons.addWidget(profile_y_button)
        buttons.addWidget(map_button)
        buttons.addWidget(compare_button)
        buttons.addStretch()

        layout.addLayout(form, 1)
        layout.addLayout(buttons)
        return page

    # ------------------------------------------------------------------
    # Apertura y preparación de controles
    # ------------------------------------------------------------------
    def open_npz(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir NPZ",
            str(Path.home()),
            "Archivos NPZ (*.npz);;Todos los archivos (*)",
        )
        if not filename:
            return

        try:
            new_npz = np.load(filename, allow_pickle=False)
            names = list(new_npz.files)
            if not names:
                new_npz.close()
                raise ValueError("El archivo NPZ no contiene variables.")

            if self.npz is not None:
                self.npz.close()

            self.npz = new_npz
            self.current_file = filename
            self.file_label.setText(filename)
            self.file_label.setToolTip(filename)
            self.setWindowTitle(f"NPZ Viewer - {os.path.basename(filename)}")
            self.export_action.setEnabled(True)

            self._fill_controls(names)
            self.variable_list.setCurrentRow(0)
            self.statusBar().showMessage(
                f"Cargadas {len(names)} variables", 5000
            )
        except Exception as exc:
            self._error("No se pudo abrir el archivo", exc)

    def _fill_controls(self, names):
        self.variable_list.clear()
        numeric_names = []

        for name in names:
            self.variable_list.addItem(name)
            try:
                array = np.asarray(self.npz[name])
                if np.issubdtype(array.dtype, np.number):
                    numeric_names.append(name)
            except Exception:
                pass

        general_combos = (
            self.xy_x_combo,
            self.xy_y_combo,
            self.contour_x_combo,
            self.contour_y_combo,
            self.contour_z_combo,
        )
        for combo in general_combos:
            combo.clear()
            combo.addItems(numeric_names)

        self.species_data_combo.clear()
        self.species_data_combo.addItems(numeric_names)

        for combo in (self.species_x_combo, self.species_ycoord_combo):
            combo.clear()
            combo.addItem("<usar índice>")
            combo.addItems(numeric_names)

        x_name = self._find_name(names, ["x", "X", "xc", "x_grid", "grid_x"])
        y_name = self._find_name(names, ["y", "ycoord", "yc", "y_grid", "grid_y", "z", "Z"])

        if x_name:
            self._set_combo(self.xy_x_combo, x_name)
            self._set_combo(self.contour_x_combo, x_name)
            self._set_combo(self.species_x_combo, x_name)

        if y_name:
            self._set_combo(self.xy_y_combo, y_name)
            self._set_combo(self.contour_y_combo, y_name)
            self._set_combo(self.species_ycoord_combo, y_name)

        species_data_name = self._find_exact_ci(names, "Y")
        if species_data_name is None:
            candidates = [
                name for name in numeric_names
                if np.asarray(self.npz[name]).ndim == 3
            ]
            species_data_name = candidates[0] if candidates else None
        if species_data_name:
            self._set_combo(self.species_data_combo, species_data_name)

        self.species_combo.clear()
        species_key = self._find_exact_ci(names, "species_names")
        if species_key:
            raw_names = np.asarray(self.npz[species_key]).ravel()
            self.species_combo.addItems(
                [self._decode_name(value) for value in raw_names]
            )

        try:
            self.species_data_combo.currentTextChanged.disconnect(
                self._update_species_dimensions
            )
        except TypeError:
            pass
        self.species_data_combo.currentTextChanged.connect(
            self._update_species_dimensions
        )
        self._update_species_dimensions()

    @staticmethod
    def _decode_name(value):
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _find_exact_ci(names, target):
        return next(
            (name for name in names if name.lower() == target.lower()),
            None,
        )

    @staticmethod
    def _find_name(names, candidates):
        for candidate in candidates:
            if candidate in names:
                return candidate
        lower_map = {name.lower(): name for name in names}
        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        return None

    @staticmethod
    def _set_combo(combo, text):
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _update_species_dimensions(self):
        if self.npz is None:
            return

        name = self.species_data_combo.currentText()
        if not name:
            return

        array = np.asarray(self.npz[name])
        if array.ndim != 3:
            self.species_nc_spin.setRange(0, 0)
            self.statusBar().showMessage(
                f"'{name}' no tiene forma (Nz, Ns, Nc): {array.shape}",
                7000,
            )
            return

        nz, ns, nc = array.shape
        self.species_nc_spin.setRange(0, max(0, nc - 1))

        if self.species_combo.count() != ns:
            self.species_combo.clear()
            self.species_combo.addItems(
                [f"Especie {k}" for k in range(ns)]
            )
            self.statusBar().showMessage(
                "El número de species_names no coincide con Ns; "
                "se utilizan nombres genéricos.",
                7000,
            )
        else:
            self.statusBar().showMessage(
                f"Y detectada: Nz={nz}, especies={ns}, Nc={nc}",
                5000,
            )

    # ------------------------------------------------------------------
    # Información y tabla
    # ------------------------------------------------------------------
    def on_variable_selected(self, name):
        if self.npz is None or not name:
            return

        try:
            array = np.asarray(self.npz[name])
            lines = [
                f"Nombre: {name}",
                f"Tipo: {array.dtype}",
                f"Dimensiones: {array.ndim}",
                f"Forma: {array.shape}",
                f"Elementos: {array.size:,}",
                f"Memoria: {array.nbytes / 1024**2:.3f} MiB",
            ]

            if np.issubdtype(array.dtype, np.number) and array.size:
                if np.issubdtype(array.dtype, np.inexact):
                    finite = array[np.isfinite(array)]
                else:
                    finite = array
                if finite.size:
                    lines.extend(
                        [
                            f"Mínimo: {np.min(finite):.10g}",
                            f"Máximo: {np.max(finite):.10g}",
                            f"Media: {np.mean(finite):.10g}",
                            f"Desviación: {np.std(finite):.10g}",
                        ]
                    )

            self.info_text.setPlainText("\n".join(lines))
            self._show_table(array)
            self.quick_label.setText(f"Variable seleccionada: {name}")
        except Exception as exc:
            self._error("No se pudo mostrar la variable", exc)

    def _show_table(self, array):
        self.table.clear()

        if array.ndim == 0:
            matrix = np.asarray([[array.item()]])
        elif array.ndim == 1:
            matrix = array.reshape(-1, 1)
        else:
            matrix = array
            while matrix.ndim > 2:
                matrix = matrix[0]

        rows = min(matrix.shape[0], MAX_TABLE_ROWS)
        columns = min(matrix.shape[1], MAX_TABLE_COLUMNS)
        self.table.setRowCount(rows)
        self.table.setColumnCount(columns)
        self.table.setHorizontalHeaderLabels(
            [str(index) for index in range(columns)]
        )

        for i in range(rows):
            for j in range(columns):
                value = matrix[i, j]
                if isinstance(value, (float, np.floating, complex, np.complexfloating)):
                    text = f"{value:.10g}"
                else:
                    text = str(value)
                self.table.setItem(i, j, QTableWidgetItem(text))

        if rows < matrix.shape[0] or columns < matrix.shape[1]:
            self.statusBar().showMessage(
                f"Tabla parcial {rows} x {columns}; forma real {matrix.shape}",
                7000,
            )

    # ------------------------------------------------------------------
    # Representaciones generales
    # ------------------------------------------------------------------
    def plot_quick_view(self):
        item = self.variable_list.currentItem()
        if self.npz is None or item is None:
            self._warning("Seleccione una variable.")
            return

        try:
            name = item.text()
            array = np.asarray(self.npz[name]).squeeze()
            ax = self.canvas.new_axes()

            if array.ndim == 0:
                ax.text(0.5, 0.5, str(array.item()), ha="center", va="center")
                ax.set_axis_off()
            elif array.ndim == 1:
                ax.plot(array)
                ax.set_xlabel("Índice")
                ax.grid(True, alpha=0.3)
            else:
                while array.ndim > 2:
                    array = array[0]
                image = ax.imshow(
                    array,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                )
                self.canvas.figure.colorbar(image, ax=ax, label=name)

            ax.set_title(name)
            self.canvas.draw_idle()
        except Exception as exc:
            self._error("Error de representación", exc)

    def plot_xy(self):
        if self.npz is None:
            self._warning("Abra primero un archivo NPZ.")
            return

        try:
            x_name = self.xy_x_combo.currentText()
            y_name = self.xy_y_combo.currentText()
            x = np.asarray(self.npz[x_name]).ravel()
            y = np.asarray(self.npz[y_name]).ravel()

            if x.size != y.size:
                raise ValueError(
                    f"Tamaños distintos: {x_name}={x.size}, {y_name}={y.size}"
                )

            finite = np.isfinite(x) & np.isfinite(y)
            style = {
                "Línea": "-",
                "Línea y marcadores": "-o",
                "Marcadores": "o",
            }[self.xy_style_combo.currentText()]

            ax = self.canvas.new_axes()
            ax.plot(x[finite], y[finite], style, markersize=3)
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(f"{y_name} frente a {x_name}")
            ax.grid(True, alpha=0.3)
            self.canvas.draw_idle()
        except Exception as exc:
            self._error("Error en gráfico XY", exc)

    def plot_contour(self):
        if self.npz is None:
            self._warning("Abra primero un archivo NPZ.")
            return

        try:
            x_name = self.contour_x_combo.currentText()
            y_name = self.contour_y_combo.currentText()
            z_name = self.contour_z_combo.currentText()

            x = np.asarray(self.npz[x_name]).squeeze()
            y = np.asarray(self.npz[y_name]).squeeze()
            z = np.asarray(self.npz[z_name]).squeeze()
            while z.ndim > 2:
                z = z[0]
            if z.ndim != 2:
                raise ValueError(f"'{z_name}' debe ser 2D; forma actual {z.shape}")

            x_grid, y_grid, z_grid = self._prepare_xy_grid(x, y, z)
            ax = self.canvas.new_axes()
            artist = self._draw_2d_artist(
                ax,
                x_grid,
                y_grid,
                z_grid,
                self.contour_type_combo.currentText(),
            )
            self.canvas.figure.colorbar(artist, ax=ax, label=z_name)
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(z_name)
            self.canvas.draw_idle()
        except Exception as exc:
            self._error("Error en contorno", exc)

    def _draw_2d_artist(self, ax, x_grid, y_grid, field, plot_type):
        cmap = self.colormap_combo.currentText()
        if plot_type == "Contorno relleno":
            return ax.contourf(
                x_grid,
                y_grid,
                field,
                levels=self.levels_spin.value(),
                cmap=cmap,
            )
        if plot_type == "Líneas":
            artist = ax.contour(
                x_grid,
                y_grid,
                field,
                levels=self.levels_spin.value(),
                cmap=cmap,
            )
            ax.clabel(artist, inline=True, fontsize=8)
            return artist
        return ax.pcolormesh(
            x_grid,
            y_grid,
            field,
            shading="auto",
            cmap=cmap,
        )

    @staticmethod
    def _prepare_xy_grid(x, y, field):
        """Normaliza coordenadas 1D/2D para un campo 2D."""
        x = np.asarray(x).squeeze()
        y = np.asarray(y).squeeze()
        field = np.asarray(field).squeeze()

        if field.ndim != 2:
            raise ValueError(f"El campo debe ser 2D; forma actual {field.shape}")

        nrows, ncols = field.shape

        if x.ndim == 1 and y.ndim == 1:
            if x.size == ncols and y.size == nrows:
                x_grid, y_grid = np.meshgrid(x, y)
                return x_grid, y_grid, field
            if x.size == nrows and y.size == ncols:
                x_grid, y_grid = np.meshgrid(x, y)
                return x_grid, y_grid, field.T
            raise ValueError(
                "Coordenadas 1D incompatibles. "
                f"X{x.shape}, Y{y.shape}, campo{field.shape}"
            )

        if x.ndim == 2 and y.ndim == 2:
            if x.shape == y.shape == field.shape:
                return x, y, field
            if x.shape == y.shape == field.T.shape:
                return x, y, field.T
            raise ValueError(
                "Las matrices X, Y y campo deben tener la misma forma. "
                f"X{x.shape}, Y{y.shape}, campo{field.shape}"
            )

        raise ValueError(
            "X e Y deben ser ambas coordenadas 1D o ambas matrices 2D. "
            f"X{x.shape}, Y{y.shape}"
        )

    # ------------------------------------------------------------------
    # Representaciones de especies Y(Nz, Ns, Nc)
    # ------------------------------------------------------------------
    def _get_species_field(self):
        if self.npz is None:
            raise ValueError("Abra primero un archivo NPZ.")

        data_name = self.species_data_combo.currentText()
        species_data = np.asarray(self.npz[data_name])
        if species_data.ndim != 3:
            raise ValueError(
                f"'{data_name}' debe tener forma (Nz, Ns, Nc); "
                f"forma actual {species_data.shape}"
            )

        nz, ns, nc = species_data.shape
        species_index = self.species_combo.currentIndex()
        if not 0 <= species_index < ns:
            raise IndexError("Índice de especie no válido")

        species_name = (
            self.species_combo.currentText()
            or f"Especie {species_index}"
        )
        field = species_data[:, species_index, :]
        return species_data, field, species_index, species_name, nz, nc

    def _get_selected_coordinate(self, combo, axis_name, nz, nc):
        selection = combo.currentText()

        if selection.startswith("<usar índice"):
            if axis_name == "x":
                return np.arange(nc, dtype=float), "Índice Nc"
            return np.arange(nz, dtype=float), "Índice Nz"

        coordinate = np.asarray(self.npz[selection]).squeeze()

        # Vectores naturales para el campo (Nz, Nc).
        if coordinate.ndim == 1:
            expected = nc if axis_name == "x" else nz
            if coordinate.size == expected:
                return coordinate, selection

            # También se acepta el vector contrario. _prepare_xy_grid puede
            # reconocer una orientación traspuesta del campo.
            alternative = nz if axis_name == "x" else nc
            if coordinate.size == alternative:
                return coordinate, selection

            raise ValueError(
                f"La coordenada '{selection}' tiene {coordinate.size} valores. "
                f"Para el eje {axis_name.upper()} se esperaban {expected}."
            )

        if coordinate.ndim == 2:
            if coordinate.shape == (nz, nc):
                return coordinate, selection
            if coordinate.shape == (nc, nz):
                return coordinate.T, selection
            raise ValueError(
                f"La coordenada '{selection}' tiene forma {coordinate.shape}; "
                f"se esperaba ({nz}, {nc}) o ({nc}, {nz})."
            )

        if coordinate.size == nz * nc:
            return coordinate.reshape(nz, nc), selection

        raise ValueError(
            f"La coordenada '{selection}' tiene forma no compatible: "
            f"{coordinate.shape}"
        )

    def _get_species_xy(self, nz, nc):
        x, x_label = self._get_selected_coordinate(
            self.species_x_combo, "x", nz, nc
        )
        y, y_label = self._get_selected_coordinate(
            self.species_ycoord_combo, "y", nz, nc
        )
        return x, y, x_label, y_label

    def plot_species_xy_map(self):
        """Representa Y[:, k, :] utilizando X e Y elegidas por el usuario."""
        try:
            _, field, k, species_name, nz, nc = self._get_species_field()
            x, y, x_label, y_label = self._get_species_xy(nz, nc)
            x_grid, y_grid, field_grid = self._prepare_xy_grid(x, y, field)

            ax = self.canvas.new_axes()
            artist = self._draw_2d_artist(
                ax,
                x_grid,
                y_grid,
                field_grid,
                self.species_plot_type_combo.currentText(),
            )
            self.canvas.figure.colorbar(
                artist,
                ax=ax,
                label=f"Y({species_name})",
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(
                f"{species_name}: Y[:, {k}, :]"
            )
            self.canvas.draw_idle()
            self.statusBar().showMessage(
                f"Mapa de {species_name}: {field_grid.shape}",
                5000,
            )
        except Exception as exc:
            self._error("Error en el mapa X-Y de la especie", exc)

    def _coordinate_profile(self, coordinate, ic, nz, nc, axis_name):
        """Extrae una coordenada de longitud Nz para un perfil Nc."""
        coordinate = np.asarray(coordinate)

        if coordinate.ndim == 2:
            if coordinate.shape != (nz, nc):
                raise ValueError(
                    f"La coordenada {axis_name} tiene forma {coordinate.shape}; "
                    f"se esperaba ({nz}, {nc})."
                )
            return coordinate[:, ic]

        if coordinate.ndim == 1:
            if coordinate.size == nz:
                return coordinate
            if coordinate.size == nc:
                # La coordenada es constante a lo largo de Nz para ese perfil.
                return np.full(nz, coordinate[ic], dtype=float)

        raise ValueError(
            f"No se puede obtener un perfil de la coordenada {axis_name}."
        )

    def plot_species_profile(self, coordinate_axis):
        """Dibuja Y[:, k, Nc] frente a la coordenada X o Y seleccionada."""
        try:
            species_data, _, k, species_name, nz, nc = self._get_species_field()
            x, y, x_label, y_label = self._get_species_xy(nz, nc)
            coordinate = x if coordinate_axis == "x" else y
            coordinate_label = x_label if coordinate_axis == "x" else y_label

            ax = self.canvas.new_axes()

            if self.species_all_nc_check.isChecked():
                for ic in range(nc):
                    coord_profile = self._coordinate_profile(
                        coordinate, ic, nz, nc, coordinate_axis.upper()
                    )
                    profile = species_data[:, k, ic]
                    finite = np.isfinite(coord_profile) & np.isfinite(profile)
                    ax.plot(
                        coord_profile[finite],
                        profile[finite],
                        linewidth=1.0,
                        alpha=0.75,
                        label=f"Nc={ic}",
                    )
                if nc <= 15:
                    ax.legend(fontsize=8, ncol=2)
                ax.set_title(
                    f"{species_name}: todos los perfiles Nc"
                )
            else:
                ic = self.species_nc_spin.value()
                coord_profile = self._coordinate_profile(
                    coordinate, ic, nz, nc, coordinate_axis.upper()
                )
                profile = species_data[:, k, ic]
                finite = np.isfinite(coord_profile) & np.isfinite(profile)
                ax.plot(
                    coord_profile[finite],
                    profile[finite],
                    linewidth=1.8,
                )
                ax.set_title(f"{species_name}, Nc={ic}")

            ax.set_xlabel(coordinate_label)
            ax.set_ylabel(f"Y({species_name})")
            ax.grid(True, alpha=0.3)
            self.canvas.draw_idle()
        except Exception as exc:
            self._error("Error al representar el perfil de especie", exc)

    def plot_all_species(self):
        """Compara todas las especies para un Nc usando la coordenada X."""
        try:
            species_data, _, _, _, nz, nc = self._get_species_field()
            x, _, x_label, _ = self._get_species_xy(nz, nc)
            ic = self.species_nc_spin.value()
            coordinate = self._coordinate_profile(x, ic, nz, nc, "X")

            ax = self.canvas.new_axes()
            ns = species_data.shape[1]
            for k in range(ns):
                species_name = (
                    self.species_combo.itemText(k)
                    if k < self.species_combo.count()
                    else f"Especie {k}"
                )
                profile = species_data[:, k, ic]
                finite = np.isfinite(coordinate) & np.isfinite(profile)
                ax.plot(
                    coordinate[finite],
                    profile[finite],
                    linewidth=1.2,
                    label=species_name,
                )

            ax.set_xlabel(x_label)
            ax.set_ylabel("Y")
            ax.set_title(f"Todas las especies, Nc={ic}")
            ax.grid(True, alpha=0.3)
            if ns <= 30:
                ax.legend(fontsize=8, ncol=2)
            self.canvas.draw_idle()
        except Exception as exc:
            self._error("Error al comparar especies", exc)

    # ------------------------------------------------------------------
    # Exportación y cierre
    # ------------------------------------------------------------------
    def export_csv(self):
        item = self.variable_list.currentItem()
        if self.npz is None or item is None:
            self._warning("Seleccione una variable.")
            return

        name = item.text()
        array = np.asarray(self.npz[name]).squeeze()
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim > 2:
            self._warning("CSV admite únicamente arrays 1D o 2D.")
            return

        default_path = Path(self.current_file).parent / f"{name}.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar CSV",
            str(default_path),
            "Archivos CSV (*.csv)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        try:
            np.savetxt(filename, array, delimiter=",", fmt="%.18g")
            self.statusBar().showMessage(
                f"CSV guardado en {filename}", 5000
            )
        except Exception as exc:
            self._error("Error al exportar", exc)

    def _warning(self, text):
        QMessageBox.warning(self, "NPZ Viewer", text)

    def _error(self, title, exception):
        QMessageBox.critical(
            self,
            title,
            f"{type(exception).__name__}: {exception}",
        )

    def closeEvent(self, event):
        if self.npz is not None:
            self.npz.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NPZ Viewer")
    app.setStyle("Fusion")
    viewer = NPZViewer()
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
