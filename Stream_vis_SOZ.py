import numpy as np
import nibabel as nib
from nilearn import datasets, plotting, image
from nilearn.image import math_img
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

class BrainZoneVisualizer:
    def __init__(self):
        self.irritative_zones = []
        self.seizure_onset_zones = []
        self.hemisphere = "Оба"  # По умолчанию оба полушария
        
        # Загрузка Harvard-Oxford cortical atlas
        print("Загрузка Harvard-Oxford Cortical Atlas...")
        self.ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        
        # В Harvard-Oxford atlas maps уже является NiftiImage объектом
        self.atlas_img = self.ho_atlas.maps
        self.atlas_data = self.atlas_img.get_fdata()
        self.atlas_labels = self.ho_atlas.labels
        
        # Загрузка стандартного мозга MNI152 для фона
        self.mni_template = datasets.load_mni152_template()
        
        # Создаем маску для полушарий
        self.create_hemisphere_masks()
        
        print(f"Загружено регионов: {len(self.atlas_labels)}")
    
    def create_hemisphere_masks(self):
        """Создает маски для левого и правого полушарий"""
        # Определяем середину мозга по оси X (левый/правый)
        mid_x = self.atlas_data.shape[0] // 2
        self.left_hemisphere_mask = np.zeros_like(self.atlas_data, dtype=bool)
        self.right_hemisphere_mask = np.zeros_like(self.atlas_data, dtype=bool)
        
        # Левое полушарие (меньшие координаты X)
        self.left_hemisphere_mask[:mid_x, :, :] = True
        # Правое полушарие (большие координаты X)
        self.right_hemisphere_mask[mid_x:, :, :] = True
        
    def create_zone_mask(self, zones, value=1):
        """Создает маску для выбранных зон с учетом полушария"""
        if not zones:
            return None
            
        mask_data = np.zeros(self.atlas_data.shape[:3])
        
        for zone_name in zones:
            # Находим индекс региона
            try:
                zone_idx = self.atlas_labels.index(zone_name)
                # Создаем маску для этого региона
                zone_mask = (self.atlas_data == zone_idx)
                
                # Применяем фильтр по полушарию
                if self.hemisphere == "Левое":
                    zone_mask = zone_mask & self.left_hemisphere_mask
                elif self.hemisphere == "Правое":
                    zone_mask = zone_mask & self.right_hemisphere_mask
                # Если "Оба", оставляем как есть
                
                mask_data[zone_mask] = value
                print(f"Добавлен регион: {zone_name} (индекс {zone_idx})")
            except ValueError:
                print(f"Регион '{zone_name}' не найден в атласе")
                continue
                
        return nib.Nifti1Image(mask_data, self.atlas_img.affine)
    
    def plot_2d_slices(self):
        """2D визуализация на срезах"""
        # Создаем маски
        irritative_mask = self.create_zone_mask(self.irritative_zones, 1)
        seizure_mask = self.create_zone_mask(self.seizure_onset_zones, 2)
        
        # Комбинированная маска
        if irritative_mask is not None and seizure_mask is not None:
            combined_mask = math_img("img1 + 2*img2", img1=irritative_mask, img2=seizure_mask)
            title = f"Ирритативная зона (красный) и Зона начала приступов (синий) - {self.hemisphere} полушарие"
            cmap = 'coolwarm'
        elif irritative_mask is not None:
            combined_mask = irritative_mask
            title = f"Ирритативная зона - {self.hemisphere} полушарие"
            cmap = 'Reds'
        elif seizure_mask is not None:
            combined_mask = seizure_mask
            title = f"Зона начала приступов - {self.hemisphere} полушарие"
            cmap = 'Blues'
        else:
            print("Не выбраны зоны для визуализации")
            return
        
        # Создаем отдельную фигуру для 2D срезов
        fig = plt.figure(figsize=(15, 5))
        
        # Визуализация в трех проекциях с использованием display_mode
        plotting.plot_roi(combined_mask, 
                         bg_img=self.mni_template,
                         cmap=cmap, 
                         alpha=0.7,
                         display_mode='ortho',  # Автоматически создает 3 проекции
                         title=title)
        
        plt.show()
    
    def plot_3d_glass_brain(self):
        """3D glass brain визуализация"""
        # Создаем маски
        irritative_mask = self.create_zone_mask(self.irritative_zones, 1)
        seizure_mask = self.create_zone_mask(self.seizure_onset_zones, 2)
        
        # Определяем какая маска будет использоваться
        if irritative_mask is not None and seizure_mask is not None:
            combined_mask = math_img("img1 + 2*img2", 
                                   img1=irritative_mask, 
                                   img2=seizure_mask)
            mask_to_plot = combined_mask
            cmap = 'coolwarm'
            title = f"Комбинированная визуализация: Ирритативная зона (красный) и Зона начала приступов (синий) - {self.hemisphere} полушарие"
        elif irritative_mask is not None:
            mask_to_plot = irritative_mask
            cmap = 'Reds'
            title = f"Ирритативная зона - {self.hemisphere} полушарие"
        elif seizure_mask is not None:
            mask_to_plot = seizure_mask
            cmap = 'Blues'
            title = f"Зона начала приступов - {self.hemisphere} полушарие"
        else:
            print("Не выбраны зоны для визуализации")
            return
        
        # Создаем фигуру для 3D визуализации
        fig = plt.figure(figsize=(16, 6))
        
        # Визуализация glass brain в двух проекциях
        plotting.plot_glass_brain(mask_to_plot, 
                                display_mode='lzr', 
                                cmap=cmap, 
                                alpha=0.7,
                                title=title,
                                figure=fig)
        
        plt.show()
    
    def plot_3d_interactive(self):
        """Интерактивная 3D визуализация"""
        irritative_mask = self.create_zone_mask(self.irritative_zones, 1)
        seizure_mask = self.create_zone_mask(self.seizure_onset_zones, 2)
        
        if irritative_mask is not None and seizure_mask is not None:
            combined_mask = math_img("img1 + 2*img2", 
                                   img1=irritative_mask, 
                                   img2=seizure_mask)
            
            view = plotting.view_img(combined_mask, 
                                   bg_img=self.mni_template,
                                   cmap='coolwarm', 
                                   opacity=0.7,
                                   title=f"3D визуализация: Ирритативная зона (красный) и Зона начала приступов (синий) - {self.hemisphere} полушарие")
            return view
            
        elif irritative_mask is not None:
            view = plotting.view_img(irritative_mask, 
                                   bg_img=self.mni_template,
                                   cmap='Reds', 
                                   opacity=0.7,
                                   title=f"3D визуализация: Ирритативная зона - {self.hemisphere} полушарие")
            return view
            
        elif seizure_mask is not None:
            view = plotting.view_img(seizure_mask, 
                                   bg_img=self.mni_template,
                                   cmap='Blues', 
                                   opacity=0.7,
                                   title=f"3D визуализация: Зона начала приступов - {self.hemisphere} полушарие")
            return view
        
        else:
            print("Не выбраны зоны для визуализации")
            return None

    def show_atlas_labels(self):
        """Показать все доступные регионы атласа"""
        print("Доступные регионы Harvard-Oxford Cortical Atlas:")
        for i, label in enumerate(self.atlas_labels):
            print(f"{i+1:2d}. {label}")

# Создаем интерфейс для выбора зон
def create_interface():
    visualizer = BrainZoneVisualizer()
    
    # Показываем доступные регионы
    visualizer.show_atlas_labels()
    
    # Виджет для выбора полушария
    hemisphere_dropdown = widgets.Dropdown(
        options=["Левое", "Правое", "Оба"],
        value="Оба",
        description='Полушарие:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    
    # Выпадающие списки для выбора зон
    irritative_dropdown = widgets.SelectMultiple(
        options=visualizer.atlas_labels,
        description='Ирритативная зона:',
        disabled=False,
        layout=widgets.Layout(width='80%', height='150px')
    )
    
    seizure_dropdown = widgets.SelectMultiple(
        options=visualizer.atlas_labels,
        description='Зона начала приступов:',
        disabled=False,
        layout=widgets.Layout(width='80%', height='150px')
    )
    
    # Кнопки для визуализации
    button_2d = widgets.Button(description="2D срезы", button_style='primary')
    button_3d = widgets.Button(description="3D Glass Brain", button_style='primary')
    button_interactive = widgets.Button(description="Интерактивная 3D", button_style='success')
    button_clear = widgets.Button(description="Очистить", button_style='warning')
    button_show_labels = widgets.Button(description="Показать регионы")
    
    output = widgets.Output()
    
    def on_hemisphere_change(change):
        visualizer.hemisphere = change['new']
        with output:
            print(f"Выбрано полушарие: {change['new']}")
    
    def on_irritative_change(change):
        visualizer.irritative_zones = list(change['new'])
        with output:
            if visualizer.irritative_zones:
                print(f"Ирритативная зона: {', '.join(visualizer.irritative_zones)}")
    
    def on_seizure_change(change):
        visualizer.seizure_onset_zones = list(change['new'])
        with output:
            if visualizer.seizure_onset_zones:
                print(f"Зона начала приступов: {', '.join(visualizer.seizure_onset_zones)}")
    
    def on_2d_click(b):
        with output:
            clear_output()
            print("Создание 2D визуализации...")
            visualizer.plot_2d_slices()
    
    def on_3d_click(b):
        with output:
            clear_output()
            print("Создание 3D Glass Brain визуализации...")
            visualizer.plot_3d_glass_brain()
    
    def on_interactive_click(b):
        with output:
            clear_output()
            print("Создание интерактивной 3D визуализации...")
            view = visualizer.plot_3d_interactive()
            if view:
                display(view)
    
    def on_clear_click(b):
        visualizer.irritative_zones = []
        visualizer.seizure_onset_zones = []
        visualizer.hemisphere = "Оба"
        irritative_dropdown.value = ()
        seizure_dropdown.value = ()
        hemisphere_dropdown.value = "Оба"
        with output:
            clear_output()
            print("Зоны очищены, полушарие сброшено на 'Оба'")
    
    def on_show_labels_click(b):
        with output:
            clear_output()
            visualizer.show_atlas_labels()
    
    # Подписываемся на события
    hemisphere_dropdown.observe(on_hemisphere_change, names='value')
    irritative_dropdown.observe(on_irritative_change, names='value')
    seizure_dropdown.observe(on_seizure_change, names='value')
    button_2d.on_click(on_2d_click)
    button_3d.on_click(on_3d_click)
    button_interactive.on_click(on_interactive_click)
    button_clear.on_click(on_clear_click)
    button_show_labels.on_click(on_show_labels_click)
    
    # Компоновка интерфейса
    controls = widgets.VBox([
        widgets.HTML("<h2> Визуализация эпилептогенных зон</h2>"),
        widgets.HTML("<h4>Harvard-Oxford Cortical Atlas</h4>"),
        hemisphere_dropdown,
        widgets.HBox([
            widgets.VBox([
                irritative_dropdown,
                widgets.HTML("<i>Области раздражения коры</i>")
            ]),
            widgets.VBox([
                seizure_dropdown,
                widgets.HTML("<i>Области начала эпилептических приступов</i>")
            ])
        ]),
        widgets.HBox([button_2d, button_3d, button_interactive, button_clear, button_show_labels])
    ])
    
    display(widgets.VBox([controls, output]))

# Пример использования с типичными эпилептогенными зонами
def example_epilepsy_zones():
    """Пример визуализации типичных эпилептогенных зон"""
    visualizer = BrainZoneVisualizer()
    
    # Типичные ирритативные зоны при височной эпилепсии
    visualizer.irritative_zones = [
        'Frontal Pole',
        'Superior Frontal Gyrus',
        'Middle Frontal Gyrus'
    ]
    
    # Типичные зоны начала приступов
    visualizer.seizure_onset_zones = [
        'Hippocampal Formation',  # Часто вовлечена при височной эпилепсии
        'Parahippocampal Gyrus, anterior division',
        'Superior Temporal Gyrus, anterior division'
    ]
    
    print("Пример: типичные эпилептогенные зоны")
    visualizer.plot_2d_slices()
    
    # Интерактивная 3D визуализация
    view = visualizer.plot_3d_interactive()
    if view:
        display(view)

# Запуск интерфейса
print("Инициализация визуализатора зон мозга...")
create_interface()
