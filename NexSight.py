import streamlit as st
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sns

st.set_page_config("NexSight")
st.markdown("<h3 style='Text-align:center;'> NexSight </h1>",unsafe_allow_html=True)
data=st.file_uploader("Upload Data ↓",['csv'])

if data is not None:
    df=pd.read_csv(data)
    st.success("File Uploaded Succesfully..")
    if st.toggle("Show Data ↓"):
        st.dataframe(df.head())
    if st.toggle("Insights ↓"):
        t1, t2, t3, t4, t5, t6, t7=st.tabs(["Statistics of Data","Data Sampling","Column's Type","Operation","Visualization","ML Model","Clean Data"])

        columns=list(df.columns)

        with t1:
            st.markdown("<h5 style='text-align:center;'>Basic info</h5>",unsafe_allow_html=True)
            rows,cols=df.shape
            st.dataframe(pd.DataFrame({"Total Rows":[f"{rows}"],"Total Columns":[f"{cols}"]}))
            st.markdown("<h5 style='text-align:center;'>Columns Names ↓</h5>",unsafe_allow_html=True)
            st.dataframe(df.columns)
            st.markdown("<h5 style='text-align:center;'>Statistics of Data</h5>",unsafe_allow_html=True)
            stat_1=df.describe()
            st.dataframe(stat_1)
            csv_1=stat_1.to_csv().encode("utf-8")
            st.download_button(label="Download Statistics of Data",file_name="Statistics.csv",data=csv_1,mime="text/csv")

        with t2:
            st.markdown("<h5 style='text-align:center;'>Data Sampling</h5>",unsafe_allow_html=True)
            if st.toggle("Single Selection"):
                row=st.number_input("Enter the Row ↓",step=1,min_value=0,max_value=rows)
                st.dataframe(df.loc[row])
                col=st.multiselect("Enter the Column ↓",columns)
                st.dataframe(df.loc[:,col])
            if st.toggle("Multi Selection"):
                row1=st.number_input("Enter the Row 1 ↓",step=1,min_value=0,max_value=rows)
                row2=st.number_input("Enter the Row 2 ↓",step=1,min_value=0,max_value=rows)
                col1=st.multiselect("Enter the Column ↓ ",columns)
                st.dataframe(df.loc[row1:row2,col1])

        with t3:
            st.markdown("<h5 style='text-align:center;'>Column's Type</h5>",unsafe_allow_html=True)
            if st.toggle("Know Column's Type"):
                t5_1=st.multiselect("Select a Column",columns,max_selections=1)
                if len(t5_1)!=0:
                    t5_2=df[t5_1[0]].dtype
                    st.success(f"{t5_2}")
            if st.toggle("Change Column's Type"):
                t5_3=st.multiselect("Select a Column to Change it's Type",columns,max_selections=1)
                if len(t5_3)!=0:
                    t5_4=st.multiselect("Select a Type",["float","int","str"],max_selections=1)
                    if len(t5_4)!=0:
                        try:
                            df[t5_3[0]]=df[t5_3[0]].astype(t5_4[0])
                            st.success(f"Column {t5_3[0]} is Changed to {t5_4[0]}")
                        except:
                            st.error("Cannot Change Type")

        with t4:
            st.markdown("<h5 style='text-align:center;'>Operation</h5>",unsafe_allow_html=True)
            if st.toggle("Group-By Operation"):
                group_1=st.multiselect("Select Columns ↓",columns)
                opera_1=st.multiselect("Select a Column for Operation",columns,max_selections=1)
                if len(group_1)!=0 and len(opera_1)!=0:
                    st.dataframe(df.groupby(by=group_1).count().reset_index())

            if st.toggle("Pivot Table"):
                p_1=st.multiselect("Select Columns ↓",columns,max_selections=1)
                p_2=st.multiselect("Select a Column for Operation",columns,max_selections=1)
                p_2_1=st.multiselect("Select a Column for index ↓",columns,max_selections=1)
                if len(p_1)!=0 and len(p_2)!=0 and len(p_2_1)!=0:
                    st.dataframe(df.pivot_table(columns=p_1[0],index=p_2_1[0],values=p_2[0]))

        with t5:
            st.markdown("<h5 style='text-align:center;'>Visualization</h5>", unsafe_allow_html=True)
            if st.toggle("CountPlot"):
                t5_1=st.multiselect("Select a Column for x - Axis ↓",columns,max_selections=1)
                t5_2=st.multiselect("Select a Column for Hue ↓",columns,max_selections=1)
                if len(t5_1)!=0:
                    fig,ax=pt.subplots(figsize=(5,3))
                    sns.countplot(data=df,x=t5_1[0],hue=t5_2[0] if len(t5_2)!=0 else None)
                    pt.grid(axis="y")
                    st.pyplot(fig)

            if st.toggle("ScatterPlot"):
                t5_1=st.multiselect("Select a Column for x - Axis ↓",columns,max_selections=1)
                t5_1_2=st.multiselect("Select a Column for y - Axis ↓",columns,max_selections=1)
                t5_2=st.multiselect("Select a Column for Hue ↓",columns,max_selections=1)
                if len(t5_1)!=0 and len(t5_1_2)!=0:
                    fig,ax=pt.subplots(figsize=(5,3))
                    sns.scatterplot(data=df,x=t5_1[0],y=t5_1_2[0],hue=t5_2[0] if len(t5_2)!=0 else None)
                    pt.grid(axis="y")
                    st.pyplot(fig)

            if st.toggle("RegPlot"):
                t5_1=st.multiselect("Select a Column for x - Axis ↓",columns,max_selections=1)
                t5_1_2=st.multiselect("Select a Column for y - Axis ↓",columns,max_selections=1)
                if len(t5_1)!=0 and len(t5_1_2)!=0:
                    fig,ax=pt.subplots(figsize=(5,3))
                    sns.regplot(data=df,x=t5_1[0],y=t5_1_2[0])
                    pt.grid(axis="y")
                    st.pyplot(fig)

        with t6:
            st.markdown("<h5 style='text-align:center;'>ML Model</h5>", unsafe_allow_html=True)
            t6_1=st.multiselect("Select a Target Variable ↓",columns,max_selections=1)
            t6_2=st.multiselect("Select Independent Variables ↓",columns,max_selections=len(columns))
            t6_Data=df.copy()

            if st.toggle("Clean Data"):
                t6_Data=t6_Data.drop_duplicates()
                t6_Data=t6_Data.interpolate()
                st.success("Data Cleaned Succesfully...")

            if len(t6_1)!=0 and len(t6_2)!=0:
                x=t6_Data[t6_2]
                y=t6_Data[t6_1[0]]

                from sklearn.preprocessing import LabelEncoder
                le=LabelEncoder()
                from sklearn.preprocessing import StandardScaler
                sc=StandardScaler()

                if st.toggle("Manual"):
                    if st.toggle("Linear Regression"):
                        Object_1=x.select_dtypes(include="object")
                        for col in Object_1:
                            x[col]=le.fit_transform(x[col])

                        from sklearn.model_selection import train_test_split
                        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

                        x_train=sc.fit_transform(x_train)
                        x_test=sc.transform(x_test)

                        st.success("Data Prepared Succesfully...")

                        if st.toggle("Train Model"):
                            from sklearn.linear_model import LinearRegression
                            reg=LinearRegression()
                            reg.fit(x_train,y_train)
                            ypred=reg.predict(x_test)

                            from sklearn.metrics import r2_score
                            st.success(f"Model Accuracy : {r2_score(y_test,ypred)}")

                            if st.toggle("Predict"):
                                user1=pd.DataFrame()
                                for col in x.columns:
                                    user1[col]=[st.number_input(f"Enter a Value for {col}")]
                                user1=sc.transform(user1)
                                st.dataframe(reg.predict(user1))

                if st.toggle("Automatic"):
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.svm import SVC,SVR
                    from sklearn.naive_bayes import GaussianNB
                    from sklearn.model_selection import GridSearchCV



                    Object_1=x.select_dtypes(include="object")
                    for col in Object_1:
                        x[col]=le.fit_transform(x[col])
                    x=sc.fit_transform(x)

                    models={
                        "LogisticRegression":{
                            "model":LogisticRegression(),
                            "params":{
                                "C":[0.1,1,10],
                                "solver":['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                                
                            }
                        },
                        "Decision Tree":{
                            "model":DecisionTreeClassifier(),
                            "params":{
                                "criterion":["gini","entropy", "log_loss"],
                                "splitter" : ["best", "random"] 
                            }
                        },
                        "Random Forest":{
                            "model":RandomForestClassifier(),
                            "params":{
                                "criterion":["gini","entropy", "log_loss"],
                                "n_estimators":[100,200,300] 
                            }
                        },
                        "SVC":{
                            "model":SVC(),
                            "params":{
                                "kernel":['linear', 'poly', 'rbf', 'sigmoid'],
                                "degree":[3,6,10,15,20],
                                "gamma":['scale', 'auto'],
                                "C":[1,10,20]
                            }
                        },
                        "Naive Bayes":{
                            "model":GaussianNB(),
                            "params":{
                                
                            }
                        }
                    }


                    
                    if st.toggle("Start Classification Training..."):
                        for model_name,val in models.items():
                                    st.success(f"Model Name :{model_name}")
                                    grid=GridSearchCV(
                                        val["model"],
                                        val["params"],
                                        cv=5,
                                        scoring="accuracy",
                                        n_jobs=-1
                                    )
                                    grid.fit(x,y)
                                    best_score=grid.best_score_
                                    best_params=grid.best_params_
                                    st.success(f"Best Accuracy :{best_score}")
                                    st.success(f"Best Attributes :{best_params}")



                    models={
                        "SVR":{
                            "model":SVR(),
                            "params":{
                                "kernel":['linear', 'poly', 'rbf', 'sigmoid'],
                                "degree":[3,6,10,15,20],
                                "gamma":['scale', 'auto'],
                                "C":[1,10,20]
                                
                            }
                        }
                    }


                    if st.toggle("Start Regression Training...."):
                        for i,val in models.items():
                                    st.success(f"Model Name : {i}")
                                    grid=GridSearchCV(
                                        val["model"],
                                        val["params"],
                                        cv=5,
                                        scoring="r2",
                                        n_jobs=-1
                                    )
                                    grid.fit(x,y)
                                    score=grid.best_score_
                                    params=grid.best_params_
                                    st.success(f"Best Accuracy : {score}")
                                    st.success(f"Best Attribute : {params}")
                                                        


        with t7:
            st.markdown("<h5 style='text-align:center;'>Clean Data</h5>", unsafe_allow_html=True)
            t7_1=df.copy()
            t7_1=t7_1.interpolate()
            t7_1=t7_1.drop_duplicates()
            if st.toggle("Download Clean Data ↓"):
                csv=t7_1.to_csv(index=False).encode("utf-8")
                st.download_button(label="Download Cleaned Data",file_name="Cleaned Data.csv",data=csv,mime="text/csv")
                st.success("Succesfully Downloaded...")
