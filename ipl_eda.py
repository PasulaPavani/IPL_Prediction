#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.metrics import DistanceMetric
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from keras.models import load_model

from sklearn.metrics import accuracy_score,classification_report
import streamlit as st

# Load your data
bat_df=pd.read_csv(r"bat_df.csv")
bat_df1=pd.read_csv(r"bat_df1.csv")
bowler_df=pd.read_csv(r"bowler_df.csv")
overs_bowled_df=pd.read_csv(r"overs_bowled_df.csv")
p_o_m=pd.read_csv(r"p_o_m.csv")
final_df=pd.read_csv(r"final_ipl_data1.csv")
final_match=pd.read_csv(r"final_match.csv")
teams_data=pd.read_csv(r"teams_data.csv")
fv= final_df.iloc[:, :-1]
cv= final_df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(fv,cv,test_size=0.20,random_state=3,stratify=cv)
#Dividing te dataset based on the type if the variables
numerical_data=x_train.select_dtypes(include=["int64","float64"])
cat_data=x_train.select_dtypes(include=["object"])
 #Pipeline to impute and Encode nominal columns
num_p=Pipeline([("imputing_n",SimpleImputer()),("scaling",StandardScaler())])
cp=Pipeline([("imputing_c",SimpleImputer(strategy="most_frequent")),("Encoder",OneHotEncoder())])
#Pipeline for column transformation to apply for different types of data
ct=ColumnTransformer([("nominal",cp,cat_data.columns),("numerical",num_p,numerical_data.columns)],remainder="passthrough")


model=Pipeline([("ct",ct),("algo",LogisticRegression(C = 10, penalty = 'l1', solver = 'liblinear'))])

model.fit(x_train,y_train)    
# Define the main function to create the dashboard
def main():
    
    
    # Add navigation sidebar
    page = st.sidebar.selectbox("Select Page", ["Home Page","Batsman Stats", "Bowler Stats","IPL_Winner_Prediction_Probability(LR)","IPL Winning Prediction Using ANN","Trophy Winners and Runner Ups", "Team History"])
    if page=="Home Page":
        import base64
        # Set background image
        @st.cache_data
        def get_img_as_base64(file):
            with open(file, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()


        img = get_img_as_base64("ipl6.jpg")
        # data:image/png;base64,{img}
        bg_image = f"""
        <style>
        [data-testid="stAppViewContainer"]  {{
        background-image: url("data:image/png;base64,{img}");
        background-size: 70% 100%;
        background-repeat: no-repeat;
        background-position: right;
        }}
        </style>
        """
        st.markdown(bg_image, unsafe_allow_html=True) 
        st.title("IPL Dashboard")
        with open('description.md', 'r') as f:
             description = f.read()
        st.markdown(description)
    # Batsman page
    elif page == "Batsman Stats":
        st.header("Batsman Stats")
        
        import base64
            # Set background image
        @st.cache_data
        def get_img_as_base64(file):
                with open(file, "rb") as f:
                    data = f.read()
                return base64.b64encode(data).decode()


        img = get_img_as_base64("bat1.jpg")
            # data:image/png;base64,{img}
        bg_image = f"""
            <style>
            [data-testid="stAppViewContainer"]  {{
            background-image: url("data:image/png;base64,{img}");
            background-size: 20% 150%;
            background-repeat: no-repeat;
            background-position : 90% 20%;
            }}
            </style>
            """      
        st.markdown(bg_image, unsafe_allow_html=True)    
        # Add dropdown to select batsman
        selected_batsman = st.selectbox("Select Batsman", bat_df['batter'].unique())
        if st.button("Submit"):
        
        # Filter data for selected batsman
            a=bat_df1[bat_df1['batter'] == selected_batsman]
        
            st.write("No. of half_centuries" ,len( a[(a["batsman_run"]>=50) &( a["batsman_run"]<100)]))
            st.write("No. of centuries" ,len( a[(a["batsman_run"]>=100) &( a["batsman_run"]<200)]))
            st.write("No. of double_centuries" ,len( a[(a["batsman_run"]>=200) &( a["batsman_run"]<300)]))

            b=bat_df[bat_df['batter'] == selected_batsman]

            st.write("Matches_played",b["Matches_played"].sum())
            st.write("Total_runs_in_IPL",b["batsman_run"].sum())

            c=p_o_m[p_o_m["Player_of_Match"]==selected_batsman]
            st.write("player_of_match",c["Count"].sum())
            if not c.empty:
                st.write("Player_of_match\n", c[["Season", "Count"]].reset_index(drop=True))
            else:
                st.write()
            #st.write("player_of_match\n",c[["Season","Count"]].reset_index(drop=True),"NA")

            plt.figure(figsize=(10, 6))
            plt.plot(b['Season'], b['batsman_run'], marker='o')
            plt.xlabel('Season')
            plt.ylabel('Runs')
            plt.title(f'Runs Across Seasons for {selected_batsman}')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        
    # Bowler page
    elif page == "Bowler Stats":
        st.header("Bowler Stats")
        import base64
        # Set background image
        @st.cache_data
        def get_img_as_base64(file):
                with open(file, "rb") as f:
                    data = f.read()
                return base64.b64encode(data).decode()


        img = get_img_as_base64("ball2.png")
            # data:image/png;base64,{img}
        bg_image = f"""
            <style>
            [data-testid="stAppViewContainer"]  {{
            background-image: url("data:image/png;base64,{img}");
            background-size: 20% 90%;
            background-repeat: no-repeat;
            background-position : 90% 60%;
            }}
            </style>
            """
        st.markdown(bg_image, unsafe_allow_html=True)
        # Add dropdown to select bowler
        selected_bowler = st.selectbox("Select Bowler", bowler_df['bowler'].unique())
        if st.button("Submit"):
         
        # Filter data for selected bowler
            b=overs_bowled_df[overs_bowled_df["bowler"]== selected_bowler]
            st.write("No. of overs bowled ",int(b["overs_bowled"].sum()))

            a=bowler_df[bowler_df["bowler"]==selected_bowler]
            st.write("matches_played",a["Matches_played"].sum())
            st.write("No. of wickets", a["isWicketDelivery"].sum())
            
            c=p_o_m[p_o_m["Player_of_Match"]==selected_bowler]
            st.write("player_of_match",c["Count"].sum())
            st.write("player_of_match\n",c[["Season","Count"]].reset_index(drop=True))
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(a['Season'], a['isWicketDelivery'], marker='o')
            plt.xlabel('Season')
            plt.ylabel('Wickets')
            plt.title(f'Wickets Across Seasons by {selected_bowler}')
            plt.xticks(rotation=45)
            st.pyplot(plt)
    elif page == "IPL_Winner_Prediction_Probability(LR)":
        


        # Streamlit app layout
        st.title('IPL Winning Prediction')
        import base64
        # Set background image
        @st.cache_data
        def get_img_as_base64(file):
            with open(file, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()


        img = get_img_as_base64("iplf.png")
        # data:image/png;base64,{img}
        bg_image = f"""
        <style>
        [data-testid="stAppViewContainer"]  {{
        background-image: url("data:image/png;base64,{img}");
        background-size: 30% 50%;
        background-repeat: no-repeat;
        background-position: right;
        }}
        </style>
        """

        st.markdown(bg_image, unsafe_allow_html=True)


        # User inputs
        # Example: user selects team statistics, player performance, match venue, etc.
        teams =['----Select----',
        'Chennai Super Kings',
        'Delhi Capitals',
        'Gujarat Titans',
        'Kolkata Knight Riders',
        'Mumbai Indians',
        'Punjab Kings',
        'Rajasthan Royals',
        'Royal Challengers Bangalore',
        'Sunrisers Hyderabad']
        
        col1, col2 = st.columns(2)

        with col1:
        
            batting_team =  st.selectbox('Select Batting Team',teams)

        with col2:
            if batting_team == '--- select ---':
                bowling_team = st.selectbox('Select Bowling Team', teams)
            else:
                filtered_teams = [team for team in teams if team != batting_team]
                bowling_team = st.selectbox('Select Bowling Team', filtered_teams)
        target = st.number_input('Target')

        col1,col2,col3 = st.columns(3)

        with col1:
            score = st.number_input('Score',step=1,format="%d",value=0)
        with col2:
            overs = st.number_input("Over Completed",step=0.1,min_value=0.0,max_value=20.0)
        with col3:
            wickets = st.number_input("wicktes down",step=1,format="%d",value=0,min_value=0,max_value=10)

        if st.button('Predict Winning Probability'):
            
                runs_left = target - score
                balls_left = 120 - (overs*6)
                wickets = 10-wickets
                crr = score/overs
                rrr = runs_left/(balls_left/6)

                input_data = pd.DataFrame({'BattingTeam':[batting_team],'BowlingTeam':[bowling_team],
                                'runs_left':[runs_left],'balls_left':[balls_left],
                                'wickets_remaining':[wickets],'target':[target],'crr':[crr],'rrr':[rrr]})
                
                result = model.predict_proba(input_data)
            
                loss = result[0][0]
                win =  result[0][1]
                st.header(batting_team + " = "+str(round(win*100)) + "%")
                st.header(bowling_team + " = "+str(round(loss*100)) + "%")
    #Wiining prediction using ANN            
    elif page == "IPL Winning Prediction Using ANN":
         # Streamlit app layout
        st.title('IPL Winning Prediction Using ANN')
        import base64
        # Set background image
        @st.cache_data
        def get_img_as_base64(file):
            with open(file, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()


        img = get_img_as_base64("iplf.png")
        # data:image/png;base64,{img}
        bg_image = f"""
        <style>
        [data-testid="stAppViewContainer"]  {{
        background-image: url("data:image/png;base64,{img}");
        background-size: 30% 50%;
        background-repeat: no-repeat;
        background-position: right;
        }}
        </style>
        """

        st.markdown(bg_image, unsafe_allow_html=True)
        best_model = load_model(r'C:\Users\LENOVO\innomatics_python\Deep_Learning\best_ipl_model_weights.keras')
        teams =['----Select----',
        'Chennai Super Kings',
        'Delhi Capitals',
        'Gujarat Titans',
        'Kolkata Knight Riders',
        'Mumbai Indians',
        'Punjab Kings',
        'Rajasthan Royals',
        'Royal Challengers Bangalore',
        'Sunrisers Hyderabad']
        
        col1, col2 = st.columns(2)

        with col1:
        
            batting_team =  st.selectbox('Select Batting Team',teams)

        with col2:
            if batting_team == '--- select ---':
                bowling_team = st.selectbox('Select Bowling Team', teams)
            else:
                filtered_teams = [team for team in teams if team != batting_team]
                bowling_team = st.selectbox('Select Bowling Team', filtered_teams)
        target = st.number_input('Target')

        col1,col2,col3 = st.columns(3)

        with col1:
            score = st.number_input('Score',step=1,format="%d",value=0)
        with col2:
            overs = st.number_input("Over Completed",step=0.1,min_value=0.0,max_value=20.0)
        with col3:
            wickets = st.number_input("wicktes down",step=1,format="%d",value=0,min_value=0,max_value=10)

        if st.button('Predict Winning Probability'):
            
                runs_left = target - score
                balls_left = 120 - (overs*6)
                wickets = 10-wickets
                crr = score/overs
                rrr = runs_left/(balls_left/6)

                input_data = pd.DataFrame({'BattingTeam':[batting_team],'BowlingTeam':[bowling_team],
                                'runs_left':[runs_left],'balls_left':[balls_left],
                                'wickets_remaining':[wickets],'target':[target],'crr':[crr],'rrr':[rrr]})
                
                x = ct.transform(input_data)
                new_data_point = x.reshape(1, -1)
                probabilities = best_model.predict(new_data_point)
                pr1 = probabilities[0][0]
                pr2 =  probabilities[0][1]
                st.header(batting_team + " = "+str(round(pr2*100)) + "%")
                st.header(bowling_team + " = "+str(round(pr1*100)) + "%")
                

    elif page == "Trophy Winners and Runner Ups":
        import base64
        # Set background image
        @st.cache_data
        def get_img_as_base64(file):
            with open(file, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()


        img = get_img_as_base64("iplf.png")
        # data:image/png;base64,{img}
        bg_image = f"""
        <style>
        [data-testid="stAppViewContainer"]  {{
        background-image: url("data:image/png;base64,{img}");
        background-size: 30% 50%;
        background-repeat: no-repeat;
        background-position: right;
        }}
        </style>
        """

        st.markdown(bg_image, unsafe_allow_html=True)
        st.write("IPL Winners and Runner Ups list")
        st.write(final_match.reset_index(drop=True))

        plt.figure(figsize=(10,6))
        sns.countplot(data=final_match,y="Winner",order=final_match["Winner"].value_counts().index)
        plt.title("Frequency of IPL Wins Finishes by Team")
        plt.xlabel("Count")
        plt.ylabel("Teams")
        st.pyplot(plt)

        plt.figure(figsize=(10,6))
        sns.countplot(data=final_match,y="Runner",order=final_match["Runner"].value_counts().index)
        plt.title("Frequency of IPL Runner-up Finishes by Team")
        plt.xlabel("Count")
        plt.ylabel("Teams")
        st.pyplot(plt)

    elif page=="Team History":
        import base64
        # Set background image
        @st.cache_data
        def get_img_as_base64(file):
            with open(file, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()


        img = get_img_as_base64("iplf.png")
        # data:image/png;base64,{img}
        bg_image = f"""
        <style>
        [data-testid="stAppViewContainer"]  {{
        background-image: url("data:image/png;base64,{img}");
        background-size: 30% 50%;
        background-repeat: no-repeat;
        background-position: right;
        }}
        </style>
        """

        st.markdown(bg_image, unsafe_allow_html=True)
        teams=['----Select Team----',
                    'Chennai Super Kings',
                    'Deccan Chargers',
                    'Delhi Capitals',
                    'Gujarat Lions',
                    'Gujarat Titans',
                    'Kochi Tuskers Kerala',
                    'Kolkata Knight Riders',
                    'Lucknow Super Giants',
                    'Mumbai Indians',
                    'Punjab Kings',
                    'Rajasthan Royals',
                    'Rising Pune Supergiant',
                    'Royal Challengers Bangalore',
                    'Sunrisers Hyderabad']   
            
        team =  st.selectbox('Select Team',teams)
        if st.button("Submit"):
        
            team_data=teams_data[teams_data["Team"]==team]
            st.write("Total_matches_played:",team_data["Matches_Played"].sum())
            st.write("Won:",team_data["Win"].sum())
            st.write("Lost:",team_data["Loss"].sum())
            tr=final_match[final_match["Winner"]==team]
            st.write("IPL Tropies: ",len(tr))
            ru=final_match[final_match["Runner"]==team]  
            st.write("IPL Runner Up : ",len(ru))      
            plt.figure(figsize=(10, 6))
            plt.plot(team_data["Season"], team_data["Win"], marker='o', label='Wins', color='green')
            plt.plot(team_data["Season"], team_data["Loss"], marker='o', label='Lost', color='red')
            plt.plot(team_data["Season"], team_data["Matches_Played"], marker='o', label='Matches Played', color='blue')
            plt.title(team+' Performance (2008-2022)')
            plt.xlabel('Year')
            plt.ylabel('Number of Matches')
            plt.xticks(team_data["Season"], rotation=45)
            plt.legend()

            st.pyplot(plt)

# Run the main function
if __name__ == "__main__":
    main()




# In[ ]:




