import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

import os
from apps import home
from myfunctions import display_animation

PWD = os.path.dirname(__file__)


# 程序入口
def main():
    # 应用页面设置（显示在网站标签）
    st.set_page_config(page_title="智能拍照服务系统", page_icon=":rainbow:", layout="wide", initial_sidebar_state="auto")

    # 初始化全局配置，加载进度页面
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
    else:
        st.session_state.first_visit = False
    if st.session_state.first_visit:
        # 在这里可以定义任意多个{***全局变量***}，方便程序进行调用
        home.web_loading()

    # 加载成功动画
    st.balloons()

    # 检测用户选择的服务状态
    with st.sidebar:
        st_lottie(display_animation(0), key="0", width=300, height=200)
        st.success("# 智能拍照服务系统\n"
                   "# Intelligent Photo Taking Service System")
        selected = option_menu("Main Menu",
                               ["主页(Home)", "智能拍照(Photographing)", "照片挑选(Selection)",
                                "图像增强(Enhancement)", "测试模块(MyTest)"],
                               icons=['house-fill', 'camera-fill', 'image-fill', 'heart-fill', 'hammer'],
                               menu_icon="robot", default_index=0)  # 注意这个default_index，默认进入显示主页
    if selected == "主页(Home)":
        home.web_home()
    if selected == "智能拍照(Photographing)":
        home.intelligent_photographing()
    if selected == "照片挑选(Selection)":
        home.photo_selection()
    if selected == "图像增强(Enhancement)":
        home.photo_enhancement()
    if selected == "测试模块(MyTest)":
        home.my_test()


if __name__ == '__main__':
    main()
