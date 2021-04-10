import streamlit as st
import streamlit.components.v1 as stc
import nltk
import re
from os import walk, path
import os
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from word2number import w2n
import plotly.express as px
from PIL import Image

import sys
import pickle
import requests

# for ocr
import pytesseract
# File Processing Pkgs
import pdfplumber


def read_pdf_with_pdfplumber(file):
    with pdfplumber.open(file) as pdf:
        page = pdf.pages[0]
        return page.extract_text()


def preprocess_pdf_text(info):
    lst_stopwords = nltk.corpus.stopwords.words("english")
    info = info.replace('\n', ' ')
    info = re.sub(r'\S+@\S+', ' ', str(info).lower())
    info = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', info, flags=re.MULTILINE)
    info = re.sub(r'[^\w\s]', ' ', str(info).lower().strip())
    info = info.split()
    info = [word for word in info if word not in lst_stopwords]
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    info = [lem.lemmatize(word) for word in info]
    return info


def load_image(image_file):
    img = Image.open(image_file)
    return img


def select_category(info):
    categories = []
    lst_stopwords = nltk.corpus.stopwords.words("english")

    for (dirpath, _, filenames) in walk(r"./keywords"):
        if filenames:
            for i in filenames:
                with open(dirpath + '/' + i) as f:
                    keyword = f.read()
                    keyword = keyword.replace('\n', ' ')
                    keyword = re.sub(r'[^\w\s]', ' ', str(
                        keyword).lower().strip())
                    keyword = keyword.split()
                    keyword = [
                        word for word in keyword if word not in lst_stopwords]
                    lem = nltk.stem.wordnet.WordNetLemmatizer()
                    keyword = [lem.lemmatize(word) for word in keyword]
                    if 'risk' in keyword:
                        keyword.remove('risk')

                    keyword = list(set(keyword))  # remove duplicate keywords

                    # type of risk can be determined from category
                    if 'category' in info:
                        idx = info.index('category')
                        if info[idx+1] in keyword or info[idx+2] in keyword or info[idx+3] in keyword or info[idx+4] in keyword or info[idx+5] in keyword or info[idx+6] in keyword:
                            categories.append(i.split('.')[0])

                    else:
                        for key in keyword:
                            if key in info:
                                categories.append(i.split('.')[0])
                                break
                f.close()

    return categories


def type_of_alert(info):
    for line in info.split('\n'):
        if line.startswith("From:"):
            if 'controlrisks' in line.lower():
                type_risk = 'CORE'

            elif 'internationalsos' in line.lower():
                type_risk = 'ISOS'

            elif 'everbridge' in line.lower():
                type_risk = 'VCC'

    try:
        return type_risk

    except:
        return None


def type_of_alert_image(info):
    lst_stopwords = nltk.corpus.stopwords.words("english")
    
    info = info.replace('\n', ' ')
    info = re.sub(r'[^\w\s]', ' ', str(info).lower().strip())
    info = info.split()
    info = [word for word in info if word not in lst_stopwords]
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    info = [lem.lemmatize(word) for word in info]

    for word in info:
        if 'controlrisks' in word or 'core' in word:
            type_risk = 'CORE'
            break

        elif 'internationalsos' in word:
            type_risk = 'ISOS'
            break

        elif 'everbridge' in word or 'vcc' in word:
            type_risk = 'VCC'
            break

    try:
        return type_risk

    except:
        return None


def extract_country_and_month_image(info):
    f = open(r"./data/countries.txt", "r",)
    for x in f:
        countries = x.lower().split(',')
    f.close()

    text = info

    lst_stopwords = nltk.corpus.stopwords.words("english")
    
    info = info.replace('\n', ' ')
    info = re.sub(r'[^\w\s]', ' ', str(info).lower().strip())
    info = info.split()
    info = [word for word in info if word not in lst_stopwords]
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    info = [lem.lemmatize(word) for word in info]

    text = text.lower()
    
    for i in countries:
        if i in text:
            country = i
            break

    for word in info:
        for i in ['april', 'august', 'december', 'february', 'january', 'july', 'june', 'march', 'may', 'november', 'october', 'september']:
            if i in word:
                month = i
                break
    try:
        return country, month

    except:
        return None


def extract_country_and_month(info):
    f = open(r"./data/countries.txt", "r",)

    for x in f:
        countries = x.lower().split(',')
    f.close()

    info = info.replace('\n', ' ')
    info = re.sub(r'[^\w\s]', ' ', str(info).lower().strip())
    info = info.split()
    info = [word for word in info if word not in lst_stopwords]
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    info = [lem.lemmatize(word) for word in info]

    for word in info:
        for i in countries:
            if i in line.lower():
                country = i

    for i in ['april', 'august', 'december', 'february', 'january', 'july', 'june', 'march', 'may', 'november', 'october', 'september']:
        if i in info:
            month = i
            break

    try:
        return country, month

    except:
        return None


def update_scores(df, info, country, month):

    df = df.reset_index()

    if 'importance' in info:
        idx = info.index('importance')
        if info[idx+1] == 'high':
            val = df.loc[(df['country'] == country) & (
                df['month'] == month), 'importance']
            df.loc[(df['country'] == country) & (
                df['month'] == month), 'importance'] = val + 0.5

    if 'level' in info:
        idx = info.index('level')
        if info[idx+1] == 'notice':
            val = df.loc[(df['country'] == country) &
                         (df['month'] == month), 'level']
            df.loc[(df['country'] == country) & (
                df['month'] == month), 'level'] = val + 0.5

        elif info[idx+1] == 'advisory':
            df.loc[(df['country'] == country) & (
                df['month'] == month), 'level']
            df.loc[(df['country'] == country) & (
                df['month'] == month), 'level'] = val + 1

    # extracting severity ['minor', 'moderate', 'severe']
    # here i will treat severity and severity score in same column
    if 'severity' in info:
        idx = info.index('severity')
        val = df.loc[(df['country'] == country) & (
            df['month'] == month), 'severity']

        if info[idx+1] == 'score':
            try:
                df.loc[(df['country'] == country) & (df['month'] == month),
                       'severity'] = val + w2n.word_to_num(info[idx+2])

            except:
                pass

        else:
            try:
                if info[idx+1] == 'minor':
                    df.loc[(df['country'] == country) & (
                        df['month'] == month), 'severity'] = val + 0.5

                elif info[idx + 1] == 'moderate':
                    df.loc[(df['country'] == country) & (
                        df['month'] == month), 'severity'] = val + 1

                elif info[idx + 1] == 'severe':
                    df.loc[(df['country'] == country) & (
                        df['month'] == month), 'severity'] = val + 2

            except:
                pass

    return df.set_index(['country', 'month'])


def update_risks(df, risks, country, month):

    df = df.reset_index()

    for risk in risks:
        val = df.loc[(df['country'] == country) & (df['month'] == month), risk]
        df.loc[(df['country'] == country) & (
            df['month'] == month), risk] = val + 1

    return df.set_index(['country', 'month'])

# for plotting the risk metre for a particular country


def plot_risk_meter(df, country, month):

    trace1 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Civil Unrest'],    domain={'row': 1, 'column': 1}, title={'text': "Civil Unrest"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Civil Unrest']}})

    trace2 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Security Risk'],    domain={'row': 1, 'column': 2}, title={'text': "Security"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Security Risk']}})

    trace3 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Crime Risk'],    domain={'row': 1, 'column': 3},    title={'text': "Crime"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Crime Risk']}})

    trace4 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Travel Risk'],    domain={'row': 1, 'column': 4},    title={'text': "Travel"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Travel Risk']}})

    trace5 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Terrorism Risk'],    domain={'row': 1, 'column': 5},    title={'text': "Terrorism"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Terrorism Risk']}})

    trace6 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Infrastructure Risk'],    domain={'row': 2, 'column': 1},    title={'text': "Infrastructure"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Infrastructure Risk']}})

    trace7 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Political Risk'],    domain={'row': 2, 'column': 2},    title={'text': "Political"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Political Risk']}})

    trace8 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Institutional Risk'],    domain={'row': 2, 'column': 3},    title={'text': "Institutional"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Institutional Risk']}})

    trace9 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Kidnap Risk'],    domain={'row': 2, 'column': 4},    title={'text': "Kidnap"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Kidnap Risk']}})

    trace10 = go.Indicator(mode="gauge+number",    value=df.loc[country, month]['Medical Risk'],    domain={'row': 2, 'column': 5},    title={'text': "Medical"}, gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'steps': [
            {'range': [0, 3], 'color': '#32CD32'},
            {'range': [3, 7], 'color': 'yellow'},
            {'range': [7, 10], 'color': 'red'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df.loc[country, month]['Medical Risk']}})

    fig = make_subplots(
        rows=2,
        cols=5,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'},
                {'type': 'indicator'}, {'type': 'indicator'},
                {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'},
                {'type': 'indicator'}, {'type': 'indicator'},
                {'type': 'indicator'}]],
    )

    fig.append_trace(trace1, row=1, col=1)
    fig.append_trace(trace2, row=1, col=2)
    fig.append_trace(trace3, row=1, col=3)
    fig.append_trace(trace4, row=1, col=4)
    fig.append_trace(trace5, row=1, col=5)
    fig.append_trace(trace6, row=2, col=1)
    fig.append_trace(trace7, row=2, col=2)
    fig.append_trace(trace8, row=2, col=3)
    fig.append_trace(trace9, row=2, col=4)
    fig.append_trace(trace10, row=2, col=5)

    st.plotly_chart(fig)


# for plotting risk trends with months
def plot_risk_trend(df, country, choice):

    months = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']

    values = []
    # as the order of months in dataframe is jumbled so has to store values in order first
    for month in months:
        values.append(df.loc[country, month][choice])

    fig = px.bar(x=months, y=values)
    fig.update_layout(
        title=country,
        xaxis_title="Months",
        yaxis_title="Values",
        showlegend=False,

        # setting arbitrary threshold for now 1.5
        shapes=[dict(
            type='line',
            line_color="#F3E40F",
            yref='y', y0=1.5, y1=1.5,
            xref='x', x0=-0.5, x1=12)],

        font=dict(
            size=18,
        )
    )

    st.plotly_chart(fig)


def azure_ocr(img):

    with open('./subscription_key.txt') as f:
        subscription_key = f.read()

    f.close()
    ocr_url = "https://loki.cognitiveservices.azure.com/" + "vision/v3.1/ocr"
    # Read the image into a byte array
    #image_data = open(image_path, "rb").read()
    # Set Content-Type to octet-stream

    img = Image.open(img)
    
    # for png files to convert the image to RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.save("./temp.jpg")

    image_data = open('./temp.jpg', "rb").read()

    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
               'Content-Type': 'application/octet-stream'}
    # put the byte array into your post request
    params = {'language': 'unk', 'detectOrientation': 'true'}
    response = requests.post(ocr_url, headers=headers,
                             params=params, data=image_data)
    response.raise_for_status()

    analysis = response.json()

    # Extract the word bounding boxes and text.
    line_infos = [region["lines"] for region in analysis["regions"]]
    word_infos = []
    data = ''
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                word_infos.append(word_info["text"])
                data = data + word_info["text"] + " "

    if os.path.exists("./temp.jpg"):
        os.remove("./temp.jpg")

    return data


def main():
    st.title("Intel Information Risk Analyzer")

    menu = ["Home", "Upload Documents", "Upload Image", "Analysis"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        country_df = pd.read_csv(r'./data/country_df.csv')
        df = px.data.gapminder().query("year == 2007")
        country_df = country_df.set_index(['country', 'month'])

        risk_avg = (country_df.groupby('country')['severity'].mean(
        ) + country_df.groupby('country')['level'].mean())/2.
        values = []
        iso_alpha = []
        for col in df['country'].values:
            for i in risk_avg.index:
                if i == col.lower():
                    values.append(risk_avg[i])
                    iso_alpha.append(df[df['country'] == col]
                                     ['iso_alpha'].values[0])

        map_df = pd.DataFrame({'iso_alpha': iso_alpha, 'values': values})

        fig = px.choropleth(map_df, locations="iso_alpha", color="values",
                            color_continuous_scale=px.colors.diverging.BrBG,
                            title="Average Risk around the world")

        st.plotly_chart(fig)

    if choice == "Upload Documents":
        df = pd.read_csv(r'./data/country_df.csv')
        st.subheader("Upload Documents")
        docx_files = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'],
                                      accept_multiple_files=True)

        if st.button("Process"):
            for docx_file in docx_files:
                if docx_file is not None:
                    file_details = {"Filename": docx_file.name,
                                    "FileType": docx_file.type, "FileSize": docx_file.size}
                    st.write(file_details)
                    # Check File Type
                    if docx_file.type == "text/plain":
                        # raw_text = docx_file.read() # read as bytes
                        # st.write(raw_text)
                        # st.text(raw_text) # fails
                        # works with st.text and st.write,used for futher processing
                        with open(docx_file) as f:
                            raw_text = f.read()
                        f.close()
                        # st.text(raw_text) # Works
                        st.write(raw_text)  # works
                        #processed_text = preprocess_pdf_text(raw_text)
                        #categories = select_category(processed_text)
                        # st.write(categories)

                    elif docx_file.type == "application/pdf":
                        #raw_text = read_pdf(docx_file)
                        # st.write(len(docx_file))
                        try:
                            df = pd.read_csv(r'./data/country_df.csv')
                            # using multindex as it would be easier to analyse and store data
                            df = df.set_index(['country', 'month'])
                            text = read_pdf_with_pdfplumber(docx_file)
                            st.write(text)
                            processed_text = preprocess_pdf_text(text)
                            #categories = select_category(processed_text)
                            categories = select_category(processed_text)
                            st.write(categories)
                            alert_type = type_of_alert(text)
                            st.write("Type of alert: " + alert_type)
                            country, month = extract_country_and_month(text)
                            st.write("Country: " + country)
                            st.write("Month: " + month)
                            df = update_risks(df, categories, country, month)
                            df = update_scores(
                                df, processed_text, country, month)
                            # for plotting risks
                            plot_risk_meter(df, country, month)
                            df.to_csv(r'./data/country_df.csv')
                        except:
                            st.write("None")

    # to view risk bar plots
    if choice == "Analysis":
        df = pd.read_csv(r'./data/country_df.csv')
        df = df.set_index(['country', 'month'])

        options = ['Civil Unrest', 'Security Risk', 'Crime Risk',
                   'Travel Risk', 'Terrorism Risk', 'Infrastructure Risk',
                   'Political Risk', 'Institutional Risk', 'Kidnap Risk', 'Medical Risk',
                   'severity', 'level', 'importance']

        countries = df.reset_index()['country'].unique()
        country = st.selectbox('Which country?', countries)
        option = st.selectbox('What Risk?', options)

        if option:
            # for plotting trend
            plot_risk_trend(df, country, option)

        option = st.selectbox('Want to view Risk Meter?', ['NO', 'Yes'])

        if option == 'Yes':
            months = ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']
            # for risk meter
            month = st.selectbox('Which Month?', months)
            plot_risk_meter(df, country, month)

    if choice == "Upload Image":
        st.subheader("Upload Image")
        image_files = st.file_uploader(
            "Upload Image", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)

        if st.button("Process"):
            for image_file in image_files:
                if image_file is not None:
                    # To See Details
                    # st.write(image_file.type)
                    # st.write(dir(image_file))
                    file_details = {"Filename": image_file.name,
                                    "FileType": image_file.type, "FileSize": image_file.size}
                    st.write(file_details)

                    img = load_image(image_file)
                    
                    # for png files to convert the image to RGB mode
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    st.image(img)

                    ocr_text = azure_ocr(image_file)

                    st.write(ocr_text)

                    df = pd.read_csv(r'./data/country_df.csv')
                    df = df.set_index(['country', 'month'])

                    try:
                        processed_text = preprocess_pdf_text(ocr_text)
                        categories = select_category(processed_text)
                        st.write(categories)
                        country, month = extract_country_and_month_image(ocr_text)
                        st.write("Country: " + country)
                        st.write("Month: " + month)
                        alert_type = type_of_alert_image(ocr_text)
                        st.write("Type of alert: " + alert_type)

                        df = update_risks(df, categories, country, month)
                        df = update_scores(df, processed_text, country, month)

                        # for plotting risks
                        plot_risk_meter(df, country, month)
                        df.to_csv(r'./data/country_df.csv')

                    except:
                        st.write("None")


if __name__ == '__main__':
    main()
