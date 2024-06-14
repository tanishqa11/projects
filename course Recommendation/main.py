import streamlit as st 
import streamlit.components.v1 as stc 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
df=pd.read_csv(r"c:\Users\Dell\Desktop\projects\course _recommend\data.csv")

def Vectorize_text_to_cosine(data):
  cv=CountVectorizer()
  cv_mat=cv.fit_transform(data)
  print(cv_mat)
  cosine_sim=cosine_similarity(cv_mat)

  return cosine_sim


def recommend(title,cosine_sim,df,num_of_rec=5):
  course_indices=pd.Series(df.index,index=df["course_title"]).drop_duplicates()
  idx=course_indices[title]
  sim_score=list(enumerate(cosine_sim[idx]))
  sim_score=sorted(sim_score,key=lambda x:x[1],reverse=True)
  select_course_indices=[i[0] for i in sim_score[1:]]
  select_course_score=[i[0] for i in sim_score[1:]]
  result_df=df.iloc[select_course_indices].copy()
  result_df.loc[:,"similarity_score"]=select_course_score
  return result_df[["course_title","similarity_score","url","price","num_subscribers"]]



RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>

</div>
"""
def main():
  st.title("Course Recommendation App")
  menu=["Home","Recommend","About"]
  choice=st.sidebar.selectbox("Menu",menu)
  if choice=="Home":
    st.subheader("Home")
    st.dataframe(df.head(10))
    st.dataframe(df.tail())


  elif choice =="Recommend":
    st.subheader("Recommend Courses")
    cosine_sim=Vectorize_text_to_cosine(df["course_title"])
    search_term=st.text_input("search")
    num_of_rec=st.sidebar.number_input("Number",4,30,7)
    if st.button("Recommend"):
      if search_term is not None:
        try:
          result=recommend(search_term,cosine_sim,df,num_of_rec)
        # st.write(result)
          for row in result.iterrows():
            rec_title=row[1].iloc[0]
            rec_score=row[1].iloc[1]
            rec_url=  row[1].iloc[2]
            rec_price=row[1].iloc[3]
            rec_num_sub =row[1].iloc[4]
            st.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_price,rec_num_sub))
        except:
          st.warning("Sorry, No Course Found")  
  else:
    st.subheader("About")
    st.text("Built with Streamlit & Pandas")
  
if __name__=="__main__":
  main()
