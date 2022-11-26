import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

import os
import datetime
from pathlib import Path

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

import base64
import os
import json
import pickle
import uuid
import re


import joblib
from numpy import random 
import cufflinks
import plotly.express as px
import time 



##GLOBAL DEFINE (INIT)
sb = st.sidebar                 #### define da sidebar
page, page_names = "", []       #### define da page & it's name
currTitle, breadcrumb = "", ""  #### define da Title & breadcrumb



def content_htmlMarkdown(varId):
    return st.markdown(varId, unsafe_allow_html=True)

def sidebar_htmlMarkdown(varId):
    return sb.markdown(varId, unsafe_allow_html=True)

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def footer_layout(*args):
    style = """
        <style>
        # MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """

    style_div = styles(
        color="#AFA1A1",
        font_style= "italic",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        text_align="center",
        height=0,
        opacity=0.6
    )

    #Hide Ori Footer
    style_hr = styles()
    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)
    content_htmlMarkdown(style)

    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)
    content_htmlMarkdown(str(foot))
    

def yearGen():   
    # Footer Copyright
    today = datetime.date.today()
    year = today.year
    activeSince = "2022"
    return year, activeSince

def footer():
    myargs = [
        "<b>Made with</b>: Python ",
        link("https://www.python.org/", image('https://i.imgur.com/ml09ccU.png',
        	width=px(18), height=px(18), margin= "0em")),
        "<b> By</b>: Hendri.",
        br(),

        "© Copyright " + str(yearGen()[0]) + " - " + str(yearGen()[1]) + ".", br(),
        "All rights reserved. Powered by: Streamlit ",
        link("https://streamlit.io/", image('https://streamlit.io/images/brand/streamlit-mark-color.png',
        	width=px(24), height=px(14), margin= "0em", align="top")),".",      
    ]
    footer_layout(*myargs)


def sideBar():
    # Website Name
    website_title = '''   
        <h1 style="margin-bottom:0; padding: 0.5rem 0px 1rem; font-size:3rem; 
        "><b><img style="width:65px; height:100%;" src="https://ars.iti.ac.id/wp-content/uploads/2022/06/Logo-ITI-oke-1-300x300.png" 
        alt="hkcode-iti"> HKCODE</b></h1> 
    '''
    sidebar_htmlMarkdown(website_title)
    
    ### re~definition
    page_names = ["🏠 Home", "🦉 About Us"]
    page = sb.radio("", page_names, index=0)
    webPage(page)


def meta():
    # Site meta data
    st.set_page_config(
        page_title="HKCode",
        page_icon="https://ars.iti.ac.id/wp-content/uploads/2022/06/Logo-ITI-oke-1-300x300.png",
        layout="wide"
    )


def webPage(page):
    if page == "🏠 Home":
        header(page)

    elif page == "🦉 About Us":
        header(page)
        content = ''' 
            <a style="font-size: 1.3rem">
                This application is made to Predict BodyFat Percentage by artificial intelligence algorithms.
                Made by: 
                <br>Name &emsp;&emsp;: Hendri 
                <br>Nim  &emsp;&ensp;&emsp;&nbsp;: 1152125001 </br>
            </a>
        '''
        content_htmlMarkdown(content)
 
 
def header(page):
    currTitle, breadcrumb = '', ''
    if page == "🏠 Home":
        welcomeText = '''   
            <h2 style="text-align: center; margin-bottom:0; padding: 0rem 1rem 0.5rem; "> \
                WELCOME to HKCODE</h2> \
            <h5 style="text-align: center; margin:0; padding: 0rem 0.5rem 1.8rem; "> \
                👌 `Body-Fat Percentage Predictor` 👌</h5>
        '''
        content_htmlMarkdown(welcomeText)
        # currTitle = '<h3 style="margin-bottom:0; padding: 0.5rem 0px 1rem;">Home</h3>'
        # content_htmlMarkdown(currTitle)

        
        subMenuOpt = st.selectbox(
            '',
            ('Predict', 'Downloads'))
        if  subMenuOpt=="Predict":
            breadcrumb = ' \
                <h6 style="margin-bottom:0; padding: 0rem 1rem 1.8rem; color:#AFA1A1;">' + page + " » " + \
                '<a style="color:#e67716;">Predict »</a></h6>'
            content_htmlMarkdown(breadcrumb)
            content('Predict')
            
        elif subMenuOpt=="Downloads":
            breadcrumb = ' \
                <h6 style="margin-bottom:0; padding: 0rem 1rem 1.8rem; color:#AFA1A1;">' + page + " » " + \
                '<a style="color:#e67716;">Downloads »</a></h6>'
            content_htmlMarkdown(breadcrumb)
            try:  
                content('Downloads')
            except ValueError:
                st.error("Streamlit is under development, some feature limited. To see how's this feature workking, download the source code.")
            pass
      
    elif page == "🦉 About Us":
        # currTitle = '<h3 style="margin-bottom:0; padding: 0.5rem 0px 1rem;">About Us</h3>'
        # content_htmlMarkdown(currTitle)
        breadcrumb = '<h6 style="margin-bottom:0; padding: 0rem 1rem 1.8rem;">' + page +' »</h6>'
        content_htmlMarkdown(breadcrumb)



def toProcessData():
    ########################################################################################
    #--------------------------------------------------------------------------------------#
    #-----------Load Comparison DataFrame/Model/Assign variables---------------------------# 
    def load_model():
        with open('./dataset/cat.pkl', 'rb') as file:
            data = joblib.load(file)
        return data
    data = load_model()
    a = {"model": data}
    regressor = a["model"] 
    df = pd.read_csv("./dataset/plotfunction.csv")
    dfe = pd.read_csv("./dataset/empty.csv")
    dfr = pd.read_csv("./dataset/randomsample.csv") 
    e = 0 
    q = df
    txt = ''

    #--------------------------------------------------------------------------------------#
    #-----------Define BodyFat Graph Functions---------------------------------------------#
    def PlotbyWeight(Weight): #All Weight Classes
        fig = px.scatter_3d(df, x='BodyFat', y='Height', z='Age', color='Weight', title=f"Body Fat % compared to others in all weight classes")
        with c1: 
            return st.plotly_chart(fig) 
    def PlotbyWeight2(q, txt): # Your Specific Weight Class Entered
        fig2 = px.scatter_3d(q, x='BodyFat', y='Height', z='Age', color='Weight', title=f"Body Fat % compared to others in weight class of {txt}")
        with c1: 
            return st.plotly_chart(fig2)
    def PlotbyWeightE(e): # Empty Default Graph at Start
        fig2 = px.scatter_3d(dfe, x='BodyFat', y='Height', z='Age', color='Weight', title="Enter some values to see some results!")
        with c1: 
            return st.plotly_chart(fig2) 

    #-----------Set Columns/Assign Inputs--------------------------------------------------#
    XRefExp1 = st.expander("", True)
    with XRefExp1:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            all = st.button("Strecth 2fit ▦")
        with c3:
            calc = st.button("Calc BodyFat %") 
        with c5: 
            rand = st.button("Random sample")
        with c4:
            reset = st.button("Reset")

        if all:
            PlotbyWeight(2)
            with c3:  
                Weight = st.number_input("Enter your Weight (Weight)")
                Height = st.number_input("Enter your Height (in)") 
                Neck = st.number_input("Enter your Neck measurement (cm)")
                Chest = st.number_input("Enter your Chest measurement (cm)") 
            with c4:    
                Hip = st.number_input("Enter your Hip measurement (cm)")
                Thigh = st.number_input("Enter your Thigh measurement (cm)")
                Knee = st.number_input("Enter your Knee measurement (cm)") 
                Ankle = st.number_input("Enter your Ankle measurement (cm)")
                Age = st.slider("Choose your age", int(dfr.Age.min()), int(dfr.Age.max())) 
            with c5: 
                Biceps = st.number_input("Enter your Biceps measurement (cm)")
                Forearm = st.number_input("Enter your Forearm measurement (cm)")
                Wrist = st.number_input("Enter your Wrist measurement (cm)")
                Abdomen = st.number_input("Enter your Waist measurement (cm)")   
        else: 
            if rand:
                w, h, n, c, hi, th, kn, an, bi, fo, wr, ag, wa = dfr.sample().values.tolist()[0]  
                with c3:
                    Weight = st.number_input("Enter your Weight (Weight)", value=w)
                    Height = st.number_input("Enter your Height (in)", value=h)
                    Neck = st.number_input("Enter your Neck measurement (cm)", value=n)
                    Chest = st.number_input("Enter your Chest measurement (cm)", value=c) 
                with c4:    
                    Hip = st.number_input("Enter your Hip measurement (cm)", value=hi)
                    Thigh = st.number_input("Enter your Thigh measurement (cm)", value=th)
                    Knee = st.number_input("Enter your Knee measurement (cm)", value=kn) 
                    Ankle = st.number_input("Enter your Ankle measurement (cm)", value=an)
                    Age = st.slider("Choose your age", int(dfr.Age.min()), int(dfr.Age.max()),   value=int(ag)) 
                with c5: 
                    Biceps = st.number_input("Enter your Biceps measurement (cm)", value=bi)
                    Forearm = st.number_input("Enter your Forearm measurement (cm)", value=fo)
                    Wrist = st.number_input("Enter your Wrist measurement (cm)", value=wr)
                    Abdomen = st.number_input("Enter your Waist measurement (cm)", value=wa)
                X = np.array([[Weight/Abdomen, Height/Abdomen, Neck/Abdomen, Chest/Abdomen, Hip/Abdomen, Thigh/Abdomen, Knee/Abdomen, Ankle/Abdomen, Biceps/Abdomen, Forearm/Abdomen, Wrist/Abdomen, Age, Abdomen]])
                X = X.astype(float)  
                percent = regressor.predict(X)
                progress_bar = st.progress(0)
                status_text = st.empty()
                if (Weight >= 125) & (Weight <= 145): 
                    q = df[(df.Weight >= 125) & (df.Weight <= 145)]
                    txt = "125-145"
                elif (Weight > 145) & (Weight <= 165):
                    q = df[(df.Weight > 145) & (df.Weight <= 165)]
                    txt = "145-165"
                elif (Weight > 165) & (Weight <= 190):
                    q = df[(df.Weight > 165) & (df.Weight <= 190)]
                    txt = "165-190"
                elif (Weight > 190) & (Weight <= 210): 
                    q = df[(df.Weight > 190) & (df.Weight <= 210)]
                    txt = "190-210"
                elif (Weight > 210) & (Weight <= 230):
                    q = df[(df.Weight > 210) & (df.Weight <= 230)]
                    txt = "210-230"
                elif (Weight > 230) & (Weight <= 250):
                    q = df[(df.Weight > 230) & (df.Weight <= 250)]
                    txt = "230-250"
                elif (Weight > 250) & (Weight <= 270): 
                    q = df[(df.Weight > 250) & (df.Weight <= 270)] 
                    txt = "250-270"
                PlotbyWeight2(q, txt)
                for i in range(100):
                    progress_bar.progress(i + 1) 
                    new_rows = np.random.randn(0,15)
                    status_text.text(f"Estimated Body Fat Percentage: {percent[0]:.2f}%")
                    time.sleep(0.00001)   
            else:
                with c3:  
                    Weight = st.number_input("Enter your Weight (Weight)")
                    Height = st.number_input("Enter your Height (in)") 
                    Neck = st.number_input("Enter your Neck measurement (cm)")
                    Chest = st.number_input("Enter your Chest measurement (cm)") 
                with c4:    
                    Hip = st.number_input("Enter your Hip measurement (cm)")
                    Thigh = st.number_input("Enter your Thigh measurement (cm)")
                    Knee = st.number_input("Enter your Knee measurement (cm)") 
                    Ankle = st.number_input("Enter your Ankle measurement (cm)")
                    Age = st.slider("Choose your age", int(dfr.Age.min()), int(dfr.Age.max()) )
                with c5: 
                    Biceps = st.number_input("Enter your Biceps measurement (cm)")
                    Forearm = st.number_input("Enter your Forearm measurement (cm)")
                    Wrist = st.number_input("Enter your Wrist measurement (cm)")
                    Abdomen = st.number_input("Enter your Waist measurement (cm)")
                if (Weight >= 125) & (Weight <= 145): 
                    q = df[(df.Weight >= 125) & (df.Weight <= 145)]
                    txt = "125-145"
                elif (Weight > 145) & (Weight <= 165):
                    q = df[(df.Weight > 145) & (df.Weight <= 165)]
                    txt = "145-165"
                elif (Weight > 165) & (Weight <= 190):
                    q = df[(df.Weight > 165) & (df.Weight <= 190)]
                    txt = "165-190"
                elif (Weight > 190) & (Weight <= 210): 
                    q = df[(df.Weight > 190) & (df.Weight <= 210)]
                    txt = "190-210"
                elif (Weight > 210) & (Weight <= 230):
                    q = df[(df.Weight > 210) & (df.Weight <= 230)]
                    txt = "210-230"
                elif (Weight > 230) & (Weight <= 250):
                    q = df[(df.Weight > 230) & (df.Weight <= 250)]
                    txt = "230-250"
                elif (Weight > 250) & (Weight <= 270): 
                    q = df[(df.Weight > 250) & (df.Weight <= 270)] 
                    txt = "250-270" 
                if calc:
                    if Weight == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1) 
                    elif Height == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1) 
                    elif Neck == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1) 
                    elif Chest == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1) 
                    elif Hip == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1) 
                    elif Thigh == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1)
                    elif Knee == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1) 
                    elif Ankle == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1) 
                    elif Age == 0: 
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1)   
                    elif Biceps == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1)
                    elif Forearm == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1)
                    elif Wrist == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1)
                    elif Abdomen == 0:
                        st.warning('Please Enter A Value Greater Than 0')
                        PlotbyWeightE(1) 
                    else:
                
                        X = np.array([[Weight/Abdomen, Height/Abdomen, Neck/Abdomen, Chest/Abdomen, Hip/Abdomen, Thigh/Abdomen, Knee/Abdomen, Ankle/Abdomen, Biceps/Abdomen, Forearm/Abdomen, Wrist/Abdomen, Age, Abdomen]])
                        X = X.astype(float)   
                        percent = regressor.predict(X)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        PlotbyWeight2(q, txt)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            new_rows = np.random.randn(0,15)
                            status_text.text(f"Estimated Body Fat Percentage: {percent[0]:.2f}%")
                            time.sleep(0.00001)
                elif all:
                    X = np.array([[Weight/Abdomen, Height/Abdomen, Neck/Abdomen, Chest/Abdomen, Hip/Abdomen, Thigh/Abdomen, Knee/Abdomen, Ankle/Abdomen, Biceps/Abdomen, Forearm/Abdomen, Wrist/Abdomen, Age, Abdomen]])
                    X = X.astype(float)
                    PlotbyWeight(1)
                    percent = regressor.predict(X)
                    progress_bar = st.progress(100)
                    status_text = st.empty()
                    new_rows = np.random.randn(0,15)
                    status_text.text(f"Estimated Body Fat Percentage: {percent[0]:.2f}%")
                    time.sleep(0.00000)   
                else:
                    PlotbyWeightE(1)




def content(ctId):
    if ctId == "Predict":          
        toProcessData()
            
    elif ctId == "Downloads":
        def download_button(pickle_it=False):
            folder_path = '.'
            filename_list = (p.resolve() for p in Path(folder_path).glob("**/*") if p.suffix in {".rar", ".py", ".txt", ".csv", ".pkl"})
            selected_filename = st.selectbox('Select a file to download', filename_list)

            fulllink = os.path.join(folder_path, selected_filename)
            with open(fulllink, 'rb') as f:
                object_to_download = f.read()

            download_filename = os.path.basename(fulllink)
            button_text = f'Download {download_filename}'
    
            if pickle_it:
                try:
                    object_to_download = pickle.dumps(object_to_download)
                except pickle.PicklingError as e:
                    st.write(e)
                    return None
            else:
                if isinstance(object_to_download, bytes):
                    pass
                elif isinstance(object_to_download, pd.DataFrame):
                    object_to_download = object_to_download.to_csv(index=False)
                # Try JSON encode for everything else
                else:
                    object_to_download = json.dumps(object_to_download)

            try:
                # some strings <-> bytes conversions necessary here
                b64 = base64.b64encode(object_to_download.encode()).decode()
            except AttributeError as e:
                b64 = base64.b64encode(object_to_download).decode()

            button_uuid = str(uuid.uuid4()).replace('-', '')
            button_id = re.sub('\d+', '', button_uuid)
            custom_css = f""" 
                <style>
                    #{button_id} {{
                        background-color: rgb(255, 255, 255);
                        color: rgb(38, 39, 48);
                        padding: 0.25em 0.38em;
                        position: relative;
                        text-decoration: none;
                        border-radius: 4px;
                        border-width: 1px;
                        border-style: solid;
                        border-color: rgb(230, 234, 241);
                        border-image: initial;
                    }} 
                    #{button_id}:hover {{
                        border-color: rgb(246, 51, 102);
                        color: rgb(246, 51, 102);
                    }}
                    #{button_id}:active {{
                        box-shadow: none;
                        background-color: rgb(246, 51, 102);
                        color: white;
                        }}
                </style> """

            dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
            return dl_link, download_filename


        try:
            download_button_str = download_button()[0]
            content_htmlMarkdown(download_button_str)
        except Exception as e:
            infoTxt = '''
                Ups Sorry, this feature is not yet supported by the streamlit library in webview.
                To see how to use this feature, plz run the source code via VSCode Terminal!
            '''
            st.info(infoTxt)

            url = "https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py"
            st.markdown("Visit sources on : [Github](%s)" % url)


if __name__ == "__main__":
    meta()
    header(page)
    content(page)
    sideBar()
    #footer()


