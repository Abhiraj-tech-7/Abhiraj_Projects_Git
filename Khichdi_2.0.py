import pandas as pd 
import streamlit as st
import mysql.connector
from twilio.rest import Client

mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    password="Momiloveu2!",
    database="Khichdi_2"
)
print(mydb)

st.set_page_config("Khichdi 2.0")
st.markdown("<h3 style='Text-align:center;'> Khichdi 2.0 </h1>",unsafe_allow_html=True)

ACCOUNT_SID = "ACd986434edddfad088757b3bbb4b5d08e"
AUTH_TOKEN = "459808af8ceceb161f6340a354e82dc0"
VERIFY_SERVICE_SID = "VAffdba406c184dcc63a51208a7e88829a"

twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

t1, t2, t3, t4 = st.tabs(["📝 Create Account", "🔐 Login", "🔄 Forgot Password", "We're Hiring!"])

with t1:
    st.subheader("📝 Create Account")
    username=st.text_input("Username ↓")
    email=st.text_input("Email ↓")
    phone=st.text_input("Phone Number ↓") 
    password=st.text_input("Password ↓", type="password")
    confirm=st.text_input("Confirm Password ↓", type="password")

    if st.button("Sign Up"):
        if not username or not email or not password or not phone:
            st.warning("Please fill all fields...")
        elif password != confirm:
            st.error("Passwords do not match!")
        else:
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="Momiloveu2!",
                database="Khichdi_2"
            )
            cur = mydb.cursor()
            cur.execute("select * from Khichdi_Users")
            data = cur.fetchall()
            cur.close()
            df = pd.DataFrame(data, columns=["username","Email","password","Phonenumber"])
            if username in df["username"].values:
                st.error("Username already exists!")
            else:
                mydb = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="Momiloveu2!",
                    database="Khichdi_2"
                )
                cur = mydb.cursor()
                sql = "insert into Khichdi_Users (username,Email,password,Phonenumber) values (%s,%s,%s,%s)"
                val = (username, email, password, phone)
                cur.execute(sql, val)
                mydb.commit()
                st.success("Account created successfully!")
                st.success("You can now 🔐 Login...")

                st.markdown(" 📱 SMS OTP Verification")
                try:
                    verification = twilio_client.verify.services(VERIFY_SERVICE_SID).verifications.create(
                        to=phone,
                        channel="sms"
                    )
                    st.success("OTP sent successfully to your phone!")
                    st.session_state["otp_phone"] = phone
                except Exception as e:
                    st.error(f"Failed to send OTP: {e}")
    if "otp_phone" in st.session_state:
        user_otp = st.text_input("Enter OTP ↓")
        if st.button("Verify OTP"):
            try:
                verification_check = twilio_client.verify.services(VERIFY_SERVICE_SID).verification_checks.create(
                    to=st.session_state["otp_phone"],
                    code=user_otp
                )
                if verification_check.status == "approved":
                    st.success("✅ Phone number verified successfully!")
                    del st.session_state["otp_phone"]
                else:
                    st.error("❌ Invalid OTP")
            except Exception as e:
                st.error(f"Error verifying OTP...")

with t2:
    st.subheader("🔐 Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("🔐 Login"):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Momiloveu2!",
            database="Khichdi_2"
        )
        cur = mydb.cursor()
        cur.execute("select * from Khichdi_Users")
        data = cur.fetchall()
        cur.close()
        df = pd.DataFrame(data, columns=["username","Email","password","Phonenumber"])
        if username in df["username"].values:
            st.success(f"Username : {username} is found!")
            if password == df.loc[df["username"] == username, "password"].iloc[0]:
                st.success(f"Welcome {username}! 🎉 You are logged in.")
            else:
                st.error(f"The Password for Username {username} is wrong!")
                st.error(f"Please try again!")
        else:
            st.error(f"Username : {username} not found...")
            st.error(f"Please Create a Account..")

with t3:
    st.subheader("🔄 Forgot Password")
    username=st.text_input(" Enter Username ↓")
    phone=st.text_input("Enter Phone Number ↓")

    if "reset_ok" not in st.session_state:
        st.session_state.reset_ok = False

    if st.button("🔄 Reset Password"):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Momiloveu2!",
            database="Khichdi_2"
        )
        cur = mydb.cursor()
        cur.execute("select * from Khichdi_Users")
        data = cur.fetchall()
        cur.close()
        df = pd.DataFrame(data, columns=["username","Email","password","Phonenumber"])

        if username in df["username"].values:
            if phone in df["Phonenumber"].values:
                if phone == df.loc[df["username"] == username, "Phonenumber"].iloc[0]:
                    st.session_state.reset_ok = True
                else:
                    st.error("Phone Number doesn't match the UserName..")
            else:
                st.error("Phone Number not Found. Please check again..")
        else:
            st.error("Username not found. Please check again.")

    if st.session_state.reset_ok:
        new_pass = st.text_input("Enter new password ↓", type="password")
        confirm = st.text_input("Confirm new password ↓", type="password")

        if st.button("Update Password"):
            if new_pass == confirm:
                mydb = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="Momiloveu2!",
                    database="Khichdi_2"
                )
                cur = mydb.cursor()
                sql = "update Khichdi_Users set password=%s where username=%s"
                val = (confirm, username)
                cur.execute(sql, val)
                mydb.commit()
                st.success("Password updated successfully!")
                st.session_state.reset_ok = False
            else:
                st.error("Passwords do not match...")

