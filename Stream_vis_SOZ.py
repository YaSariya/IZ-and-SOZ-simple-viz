import streamlit as st
import numpy as np
import nibabel as nib
from nilearn import datasets, plotting, image
from nilearn.image import math_img
import matplotlib.pyplot as plt
import tempfile
import os

# Кэшируем загрузку данных для производительности
@st.cache_resource
def load_brain_data():
    """Загрузка данных атласа и шаблона мозга"""
    st.info("Загрузка Harvard-Oxford Cortical Atlas...")
    ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    mni_template = datasets.load_mni152_template()
    return ho_atlas, mni_template

class BrainZoneVisualizer:
    def __init__(self, ho_atlas, mni_template):
        self.irritative_zones = []
        self.seizure_onset_zones = []
        self.hemisphere = "Оба"  # По умолчанию оба полушария
        
        # В Harvard-Oxford atlas maps уже является NiftiImage объектом
        self.atlas_img = ho_atlas.maps
        self.atlas_data = self.atlas_img.get_fdata()
        self.atlas_labels = ho_atlas.labels
        self.mni_template = mni_template
        
        # Создаем маску для полушарий
        self.create_hemisphere_masks()
        
        st.success(f"Загружено регионов: {len(self.atlas_labels)}")
    
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
                st.write(f"Добавлен регион: {zone_name}")
            except ValueError:
                st.warning(f"Регион '{zone_name}' не найден в атласе")
                continue
                
        return nib.Nifti1Image(mask_data, self.atlas_img.affine)
    
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
            title = f"SOZ & IZ"
        elif irritative_mask is not None:
            mask_to_plot = irritative_mask
            cmap = 'Reds'
            title = f"IZ"
        elif seizure_mask is not None:
            mask_to_plot = seizure_mask
            cmap = 'Blues'
            title = f"SOZ"
        else:
            st.warning("Не выбраны зоны для визуализации")
            return
        
        # Создаем фигуру для 3D визуализации
        fig = plt.figure(figsize=(16, 6))
        
        # Визуализация glass brain
        plotting.plot_glass_brain(mask_to_plot, 
                                display_mode='lzr', 
                                cmap=cmap, 
                                alpha=0.7,
                                title=title,
                                figure=fig)
        
        st.pyplot(fig)
        plt.close(fig)
    
    def plot_3d_interactive(self):
        """Интерактивная 3D визуализация - сохраняем как HTML"""
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
                                   title=f"SOZ & IZ")
            
        elif irritative_mask is not None:
            view = plotting.view_img(irritative_mask, 
                                   bg_img=self.mni_template,
                                   cmap='Reds', 
                                   opacity=0.7,
                                   title=f"IZ")
            
        elif seizure_mask is not None:
            view = plotting.view_img(seizure_mask, 
                                   bg_img=self.mni_template,
                                   cmap='Blues', 
                                   opacity=0.7,
                                   title=f"3SOZ")
        
        else:
            st.warning("Не выбраны зоны для визуализации")
            return None
        
        # Сохраняем как временный HTML файл
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            view.save_as_html(tmp_file.name)
            return tmp_file.name

    def show_atlas_labels(self):
        """Показать все доступные регионы атласа"""
        st.subheader("Доступные регионы Harvard-Oxford Cortical Atlas:")
        for i, label in enumerate(self.atlas_labels):
            st.write(f"{i+1:2d}. {label}")

def main():
    st.set_page_config(
        page_title="Визуализация эпилептогенных зон",
        page_icon="",
        layout="wide"
    )
    
    st.title("Визуализация эпилептогенных зон")
    st.markdown("### Harvard-Oxford Cortical Atlas")
    
    # Загрузка данных
    ho_atlas, mni_template = load_brain_data()
    visualizer = BrainZoneVisualizer(ho_atlas, mni_template)
    
    # Инициализация состояния сессии
    if 'irritative_zones' not in st.session_state:
        st.session_state.irritative_zones = []
    if 'seizure_onset_zones' not in st.session_state:
        st.session_state.seizure_onset_zones = []
    if 'hemisphere' not in st.session_state:
        st.session_state.hemisphere = "Оба"
    
    # Боковая панель для выбора зон
    with st.sidebar:
        st.header("Выбор зон")
        
        # Выбор полушария
        hemisphere = st.radio(
            "Выберите полушарие:",
            options=["Левое", "Правое", "Оба"],
            index=2,  # По умолчанию "Оба"
            help="Выберите полушарие для визуализации"
        )
        
        visualizer.hemisphere = hemisphere
        st.session_state.hemisphere = hemisphere
        
        # Выбор ирритативных зон
        irritative_selected = st.multiselect(
            "Ирритативная зона:",
            options=visualizer.atlas_labels,
            default=st.session_state.irritative_zones,
            help="Области раздражения коры"
        )
        
        # Выбор зон начала приступов
        seizure_selected = st.multiselect(
            "Зона начала приступов:",
            options=visualizer.atlas_labels,
            default=st.session_state.seizure_onset_zones,
            help="Области начала эпилептических приступов"
        )
        
        # Кнопки управления
        col1, col2 = st.columns(2)
        with col1:
            show_labels = st.button("📋 Показать регионы")
        with col2:
            clear_all = st.button("🗑️ Очистить всё")
        
        # Кнопки визуализации
        st.markdown("---")
        st.subheader("Визуализация")
        col1, col2 = st.columns(2)
        with col1:
            plot_3d = st.button("🧊 3D Glass Brain")
        with col2:
            plot_interactive = st.button("🎮 Интерактивная 3D")
    
    # Обновление состояния
    visualizer.irritative_zones = irritative_selected
    visualizer.seizure_onset_zones = seizure_selected
    st.session_state.irritative_zones = irritative_selected
    st.session_state.seizure_onset_zones = seizure_selected
    
    # Основная область контента
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Выбранные зоны")
        st.write(f"**Полушарие:** {hemisphere}")
        
        if visualizer.irritative_zones:
            st.write("**Ирритативная зона:**")
            for zone in visualizer.irritative_zones:
                st.write(f"- {zone}")
        
        if visualizer.seizure_onset_zones:
            st.write("**Зона начала приступов:**")
            for zone in visualizer.seizure_onset_zones:
                st.write(f"- {zone}")
    
    with col2:
        st.subheader("Статус")
        if not visualizer.irritative_zones and not visualizer.seizure_onset_zones:
            st.info("Выберите зоны для визуализации в боковой панели")
        else:
            st.success(f"Выбрано зон: {len(visualizer.irritative_zones) + len(visualizer.seizure_onset_zones)}")
    
    # Обработка действий
    if show_labels:
        st.subheader("Все доступные регионы")
        visualizer.show_atlas_labels()
    
    if clear_all:
        st.session_state.irritative_zones = []
        st.session_state.seizure_onset_zones = []
        st.session_state.hemisphere = "Оба"
        st.rerun()
    
    if plot_3d:
        st.subheader("3D Glass Brain")
        visualizer.plot_3d_glass_brain()
    
    if plot_interactive:
        st.subheader("Интерактивная 3D визуализация")
        st.info("Генерируется интерактивная 3D модель... Это может занять несколько секунд.")
        
        html_file = visualizer.plot_3d_interactive()
        if html_file:
            # Читаем HTML файл и отображаем его
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Отображаем HTML контент
            st.components.v1.html(html_content, height=600, scrolling=True)
            
            # Предлагаем скачать HTML файл
            with open(html_file, 'rb') as f:
                st.download_button(
                    label="💾 Скачать интерактивную 3D модель",
                    data=f,
                    file_name="brain_3d_visualization.html",
                    mime="text/html"
                )
            
            # Удаляем временный файл
            os.unlink(html_file)

if __name__ == "__main__":
    main()
