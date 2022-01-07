import streamlit as st

from myfunctions import *
import xcc_ps.al_select as xcc
import numpy
import pandas
import time
import os

from streamlit.elements.image import image_to_url
from streamlit_lottie import st_lottie

PWD = os.path.dirname(os.path.dirname(__file__))


def background():
    photo_path = os.path.join(PWD, 'test/images/mountain.jpg')
    # 加载背景图
    img_url = image_to_url(photo_path, width=1280, clamp=False, channels='RGB', output_format='auto',
                           image_id='', allow_emoji=False)
    st.markdown('''
    <style>
    .css-fg4pbf {background-image: url(''' + img_url + ''');}</style>
    ''', unsafe_allow_html=True)


# 网页加载等待页面
def web_loading():
    placeholder1, placeholder2, placeholder3 = st.empty(), st.empty(), st.empty()
    placeholder1.info("正在加载组件，请稍候...")
    bar = placeholder2.progress(0)
    for percentage in range(5):
        bar.progress(percentage * 25)
        time.sleep(0.1)
    placeholder3.success('Done!')
    time.sleep(0.5)
    placeholder1.empty()
    placeholder2.empty()
    placeholder3.empty()


# 返回主页
def web_home():
    st.warning("# 欢迎来到智能拍照服务系统！\n"
               "# Welcome to the Intelligent Photo Taking Service System")
    st.caption("### (基于树莓派的)")
    st.markdown('''---''')  # 分割线
    # 显示当前时间
    show_datetime(0)
    p11, p12 = st.columns([3, 2])
    map_data = pandas.DataFrame(numpy.random.randn(1, 2) / [100, 100] + [30.516798, 114.363390],
                                columns=['lat', 'lon'])
    with p11:
        st.map(map_data)
    with p12:
        st.image(show_image("test/images/pi.png"))
        st.image(show_image("test/images/mypi.jpg"), width=450)


# 智能拍照
def intelligent_photographing():
    c1, c3 = st.columns(2)
    with c1:
        # 网页标题
        st.warning(web_title("IP")['t'])
        st.caption(web_title("IP")['c'])
        st.markdown('''---''')  # 分割线
    with c3:
        st_lottie(display_animation(1), key="1", width=750, height=225)
    # 显示当前时间
    show_datetime(1)
    st.markdown('''---''')  # 分割线
    col1, col2 = st.columns(2)
    with col1:
        st.header("树莓派实时推流")
        placeholder = st.empty()
        if placeholder.button(label="开启推流"):
            placeholder.empty()
            # st.image(show_image("test/images/background.png"), caption='This is a test', width=800)
            # st.video(data=show_video("test/videos/02.mp4"), format='video/mp4')
            # display_url_video('http://192.168.1.106:8080/?action=stream', 0)
            display_url_video('https://www.bilibili.com/', 0)
    with col2:
        # 上传多个文件 Upload multi files
        st.info("上传文件")
        up_loaded_files = st.file_uploader(label="请选择上传文件（upload）", accept_multiple_files=True,
                                           help="选择您想要上传的文件")
        if st.button("开始上传文件"):
            upload_file(up_loaded_files)

        # 下载单个文件
        st.info("下载文件")
        down_loaded_files = st.file_uploader(label="请选择下载文件（download）", accept_multiple_files=False,
                                             help="选择您想要下载的文件")
        if st.button("确认"):
            download_file(down_loaded_files)


def photo_selection():
    c1, c3 = st.columns(2)
    with c1:
        # 网页标题
        st.warning(web_title("PS")['t'])
        st.caption(web_title("PS")['c'])
        st.markdown('''---''')  # 分割线
    with c3:
        st_lottie(display_animation(2), key="1", width=750, height=225)
    # 显示当前时间
    show_datetime(1)
    st.markdown('''---''')  # 分割线
    # 展示结果
    if st.button(label="开始挑选(Go Selection)"):
        # 计算等待
        placeholder = st.empty()
        placeholder.info("#### 正在计算中...")
        # xcc挑选算法
        xcc.my_ps()
        # 计算完成
        placeholder.success('#### Done!')
        time.sleep(1)
        placeholder.empty()
        # 展示结果
        t11, t12 = st.columns(2)
        p11, p12 = st.columns([1, 1])
        t11.info("#### 输入图像")
        t12.info("#### 最佳图像")
        # 原始图片路径
        ps_path = os.path.join(PWD, "xcc_ps/data/61")
        ls = os.listdir(ps_path)
        c_path = []
        for i in ls:
            c_path.append(os.path.join(ps_path, i))
        # 原始输入图片
        with t11:
            for x in c_path:
                p11.image(show_select_image(x), width=450)
        # 最佳图像
        # 增强后图片路径
        psed_path = os.path.join(PWD, "xcc_ps/output")
        ls1 = os.listdir(psed_path)
        c1_path = []
        for i in ls1:
            c1_path.append(os.path.join(psed_path, i))
        with t12:
            for x in c1_path:
                p12.image(show_select_image(x), width=450)


# 图像增强
def photo_enhancement():
    c1, c3 = st.columns(2)
    with c1:
        # 网页标题
        st.warning(web_title("PE")['t'])
        st.caption(web_title("PE")['c'])
        st.markdown('''---''')  # 分割线
    with c3:
        st_lottie(display_animation(3), key="1", width=750, height=225)
    # 显示当前时间
    show_datetime(1)
    st.markdown('''---''')  # 分割线
    st.error("#### 该功能尚在调试中")
    # 上传多个文件 Upload multi files
    st.info("上传文件")
    up_loaded_files = st.file_uploader(label="请选择上传文件（upload）", accept_multiple_files=False,
                                       help="选择您想要上传的文件")
    if st.button(label="开始增强(Go Enhancement)"):
        if up_loaded_files:
            # 计算等待
            placeholder = st.empty()
            with st.spinner('### 计算中...'):
                # 这里以后加上运行命令
                time.sleep(1)
            placeholder.success('Done!')
            placeholder.empty()
            c1, c2 = st.columns(2)
            c1.info("# Before")
            c2.info("# After")
            st.image("dped/output/pic1.png", caption='This is a test')
        else:
            st.error("#### 您还未选择文件")


def my_test():
    c1, c3 = st.columns(2)
    with c1:
        # 网页标题
        st.warning(web_title("TEST")['t'])
        st.caption(web_title("TEST")['c'])
        st.markdown('''---''')  # 分割线
    with c3:
        st_lottie(display_animation(4), key="1", width=750, height=225)
    # 显示当前时间
    show_datetime(1)
    st.markdown('''---''')  # 分割线
    c11, c12 = st.columns([1, 3])
    with c11:
        st.info("### 在这里输入您的链接:")
    with c12:
        url = c12.text_input("输入链接")
    if url:
        display_url_video(url, 1)
        # display_url_video('https://www.bilibili.com/', 1)
