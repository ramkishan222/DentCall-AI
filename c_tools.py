from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
import time

# # Load environment variables from .env file
load_dotenv()
x_api_key = os.getenv('X-API-KEY')
user_info = os.getenv('USER_INFO')
user_update = os.getenv('USER_UPDATE')
clinic_api = os.getenv('CLINIC_INFO')
appointment_book = os.getenv('APPOINTMENT_BOOKING')
appointment_update = os.getenv('APPOINTMENT_CHANGE')
get_policy = os.getenv('GET_POLICY')
get_avail_slots = os.getenv('AVAIL_SLOTS')



def get_policies(api_key):
    url = get_policy
    headers = {
        "x-api-key": api_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()  # Return the policies as JSON
    else:
        raise Exception(f"Error fetching policies: {response.status_code}, {response.text}")

def load_embaddings():
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    return hf

def load_retriver():
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=True,
        )
    doc = text_splitter.create_documents([policy['data']['cancellationPolicy'] ,policy['data']['paymentPolicy']])
    embad = load_embaddings()
    vector_store = Chroma.from_documents(doc , embad)
    retriver = vector_store.as_retriever(k=1)
    return retriver

policy = get_policies(x_api_key)
retriever = load_retriver()

def patient_id(phone_number: str, clinic_id: str):
    """Retrieve the patient ID based on phone number and clinic ID."""
    data = {'contact': phone_number}
    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
    try:
        response = requests.post(f'{user_info}/{clinic_id}', json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result['message']['_id']
        elif response.status_code == 404:
            return {"status": "error", "message": "User not found", "code": 404}
        else:
            return {"status": "error", "message": f"Error fetching patientId, received {response.status_code} from server"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}


def appointment_id(home_phone, clinic_id, appointment_time):
    data = {'contact': home_phone}
    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}

    try:
        response = requests.post(f'{user_info}/{clinic_id}', json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            appointments = result['message']['appointment']

            # Parse the input appointment_time to datetime object (ISO format)
            input_time = datetime.strptime(appointment_time, '%Y-%m-%dT%H:%M')

            # Iterate over the appointments and find a match
            for appointment in appointments:
                appointment_str = appointment['appointment_time']

                # Try parsing with various formats, including seconds if present
                try:
                    appointment_time_obj = datetime.strptime(appointment_str, '%Y-%m-%dT%H:%M:%S')
                except ValueError:
                    try:
                        appointment_time_obj = datetime.strptime(appointment_str, '%Y-%m-%dT%H:%M')
                    except ValueError:
                        try:
                            appointment_time_obj = datetime.strptime(appointment_str, '%Y-%m-%d %H:%M')
                        except ValueError:
                            appointment_time_obj = datetime.strptime(appointment_str, '%Y-%m-%d %I:%M %p')

                # Compare both times (ignore seconds)
                if appointment_time_obj.replace(second=0) == input_time:
                    return appointment['_id']  # Return only the _id of the matched appointment

            return {"status": "error", "message": "No matching appointment found."}

        else:
            return {"status": "error", "message": f"Failed to retrieve appointments. Received {response.status_code}"}

    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}

@tool
def lookup_policy(user_input: str) -> str:
    """Consult the company policies to check whether certain options are permitted."""
    s_time = time.time()
    docs = retriever.get_relevant_documents(user_input)
    print(f'Total time is taken in lookup policy {time.time()-s_time} \n')
    return docs #"\n\n".join([doc["page_content"] for doc in docs])

@tool
def fetch_clinic_information(config: RunnableConfig) -> str:
    """Fetch all corresponding clinic information and appointments details."""
    clinic_id = config.get('configurable', {}).get('clinic_id', None)
    if not clinic_id:
        raise ValueError("No clinic_id configured.")
    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
    try:
        s_time = time.time()
        response = requests.get(f'{clinic_api}/{clinic_id}', headers=headers)
        if response.status_code == 200:
            print(f'Total time is taken in fetch clinic info {time.time()-s_time} \n')
            return response.json()

        else:
            return None
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

@tool
def user_exists(config: RunnableConfig) -> str:
    """Check if the user exists based on their contact number."""
    clinic_id = config.get('configurable', {}).get('clinic_id', None)
    phone_number = config.get('configurable', {}).get('home_phone', None)
    if not clinic_id or not phone_number:
        raise ValueError("No clinic_id or phone number configured.")
    data = {'contact': phone_number}
    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
    try:
        s_time = time.time()
        response = requests.post(f'{user_info}/{clinic_id}', json=data, headers=headers)
        if response.status_code == 200:
            print(f'Total time is taken in user existing {time.time()-s_time} \n')
            return {"status": "success", "message": "User Exists", "response": response.json()}

        else:
            return {"status": "failure", "message": "User does not exist in our clinic", "status": response.status_code, "response": response.json()}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}

@tool
def user_verification(dob: str, config: RunnableConfig) -> str:
    """Check user verification based on their date of birth."""
    clinic_id = config.get('configurable', {}).get('clinic_id', None)
    phone_number = config.get('configurable', {}).get('home_phone', None)
    if not clinic_id or not phone_number:
        raise ValueError("No clinic_id or phone number configured.")
    data = {'contact': phone_number}
    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
    try:
        s_time = time.time()
        response = requests.post(f'{user_info}/{clinic_id}', json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print('User is Exist in clinic')
            if result['message']['dob'] == dob:
                print(f'Total time is taken dob verification {time.time()-s_time} \n')
                return {"status": "success", "message": "User Verified Successfully", "response": result}
                
            else:
                return {"status": "success", "message": "User not Verified Successfully update details", "response": result}
        else:
            return {"status": "failure", "message": 'User is not Exist register first'}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}

@tool
def book_appointment(AptDateTime: str, config: RunnableConfig) -> str:
    """Book an appointment in the database."""
    phone_number = config.get('configurable', {}).get('home_phone', None)
    clinic_id = config.get('configurable', {}).get('clinic_id', None)
    if not phone_number or not clinic_id:
        return {"status": "error", "message": "No phone number or clinic id configured."}
    s_time = time.time()
    patientId = patient_id(phone_number, clinic_id)
    if isinstance(patientId, dict) and patientId.get("status") == "error":
        return patientId
    print(patientId)
    data = {'patientId': patientId, 'AptDateTime': AptDateTime}
    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
    try:
        response = requests.post(f'{appointment_book}/{clinic_id}', json=data, headers=headers)
        if response.status_code == 201:
            result = response.json()
            print(f'Total time is taken in book appointment {time.time()-s_time} \n')
            return {"status": "success", "message": "Appointment Booked Successfully", "response": result}
        elif response.status_code == 400:
            error_message = response.json()

            return {"status": "error", "message": f"Bad Request: {error_message}"}

        else:
            return {"status": "error", "message": f"Received {response.status_code} from server."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}

@tool
def update_user_details(email: str, first_name: str, last_name: str, dob: str, config: RunnableConfig) -> str:
    """Update user details in the database."""
    clinic_id = config.get('configurable', {}).get('clinic_id', None)
    phone_number = config.get('configurable', {}).get('home_phone', None)
    if not clinic_id or not phone_number:
        raise ValueError("No clinic_id or phone number configured.")
    s_time = time.time()
    patientId = patient_id(phone_number, clinic_id)
    if isinstance(patientId, dict) and patientId.get("status") == "error":
        return patientId

    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
    payload = {'email': email, 'first_name': first_name, 'last_name': last_name, 'contact': phone_number, 'dob': dob, 'clinic_id': clinic_id}
    try:
        response = requests.patch(f'{user_update}/{patientId}', json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f'Total time is taken {time.time()-s_time} \n')
            return {"status": "success", "message": "User Profile Updated Successfully", "response": result}
        else:
            return {"status": "error", "message": f"Received {response.status_code} from server."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}

@tool
def register_new_user(first_name: str, last_name: str, gender: str, email: str, dob: str, config: RunnableConfig) -> str:
    """Register a new user in the database."""
    clinic_id = config.get('configurable', {}).get('clinic_id', None)
    phone_number = config.get('configurable', {}).get('home_phone', None)
    if not clinic_id or not phone_number:
        raise ValueError("No clinic_id or phone number configured.")

    data = {'first_name': first_name, 'last_name': last_name, 'gender': gender, 'email': email, 'dob': dob, 'contact': phone_number, 'clinic_id': clinic_id}
    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
    try:
        response = requests.post(f'{user_info}/{clinic_id}', json=data, headers=headers)
        if response.status_code == 201:
            result = response.json()
            return {"status": "success", "message": "User Registered Successfully", "response": result}
        else:
            return {"status": "error", "message": f"Received {response.status_code} from server."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}


# ## Check Appointment
# @tool
# def Check_available_slots(date:str ,config:RunnableConfig)-> str:
#     """To Check the available slots for appointment according to date"""
#     clinic_id = config.get('configurable', {}).get('clinic_id',None)
#     if not clinic_id:
#         raise ValueError('Clinic Id is not Configured')
#     try:
#         s_time = time.time()
#         response = requests.get(f'{get_avail_slots}/{clinic_id}?date={date}' , headers={'x-api-key' : x_api_key})
#         if response.status_code == 200:
#             result = response.json()
#             # Log the full response for debugging
#             print(f'Total time is taken {time.time()-s_time} \n')
#             return {"status": "sucess", "message": "Available Slots", "response": result}

#         else:
#             return {"status": "error", "message": f"Received {response.status_code} from server."}

#     except requests.exceptions.RequestException as e:
#         return {"status": "error", "message": f"Request failed: {str(e)}"}


## Check Appointment
@tool
def Check_available_slots(date: str, config: RunnableConfig) -> str:
    """Checks available appointment slots for a given date."""
    clinic_id = config.get('configurable', {}).get('clinic_id', None)
    if not clinic_id:
        raise ValueError("Clinic ID is not configured.")
    
    # Validate date
    try:
        input_date = datetime.strptime(date, "%Y-%m-%d")
        if input_date < datetime.now():
            return {"status": "error", "message": "you can not book on past dates."}
    except ValueError:
        return {"status": "error", "message": "Invalid date format. Use YYYY-MM-DD."}
    
    try:
        start_time = time.time()
        response = requests.get(
            f"{get_avail_slots}/{clinic_id}?date={date}",
            headers={"x-api-key": x_api_key}
        )
        
        elapsed_time = time.time() - start_time
        print(f"Request completed in {elapsed_time} seconds.")
        
        if response.status_code == 200:
            result = response.json()
            return {"status": "success", "message": "Available Slots", "response": result}
        else:
            return {"status": "error", "message": f"Received {response.status_code} from server."}
    
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}



## Cancle and Rescedule Appointment tool
@tool
def cancel_appointment(appointment_time:str, config: RunnableConfig) -> str:
    """Cancel an appointment in the database."""
    clinic_id = config.get('configurable', {}).get('clinic_id', None)
    phone_number = config.get('configurable', {}).get('home_phone', None)
    if not clinic_id:
        raise ValueError("No clinic_id configured.")

    if not phone_number:
        raise ValueError('No Phone Number Configured.')
    # Get appointment ID
    appoint_id = appointment_id(phone_number , clinic_id ,appointment_time)
    if isinstance(appoint_id, dict) and appoint_id.get("status") == "error":
        print(f'appointment_id is : {appoint_id}')
        return appoint_id  # Return the error from patient_id function
    # Get patient ID
    patientId = patient_id(phone_number , clinic_id)
    if isinstance(patientId, dict) and patientId.get("status") == "error":
        print(f'patientId is : {patientId}')
        return patientId  # Return the error from patient_id function

    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
    payload = {'patientId': patientId, 'clinicId': clinic_id ,  "appointment_status": "cancelled"}
    try:
        response = requests.put(f'{appointment_update}/{appoint_id}', json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            # Log the full response for debugging
            return {"status": "sucess", "message": "Appointment Cancelled Successfully", "response": result}
        else:
            return {"status": "error", "message": f"Received {response.status_code} from server."}

    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}

@tool
def reschedule_appointment(old_appointment_time :str , new_appointment_time:str, config: RunnableConfig) -> str:
    """Reschedule an appointment in the database."""
    clinic_id = config.get('configurable', {}).get('clinic_id', None)
    phone_number = config.get('configurable', {}).get('home_phone', None)
    if not clinic_id:
        raise ValueError("No clinic_id configured.")

    if not phone_number:
        raise ValueError('No Phone Number Configured.')

    # Get patient ID
    patientId = patient_id(phone_number, clinic_id)
    if isinstance(patientId, dict) and patientId.get("status") == "error":
        return patientId  # Return the error from patient_id function
    # Get appointment ID
    appoint_id = appointment_id(phone_number , clinic_id ,old_appointment_time)
    if isinstance(appoint_id, dict) and appoint_id.get("status") == "error":
        return appoint_id  # Return the error from patient_id function

    headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}
    payload = {'patientId': patientId, 'clinicId': clinic_id, 'appointment_time': new_appointment_time , "appointment_status": "not-visit"}
    try:
        response = requests.put(f'{appointment_update}/{appoint_id}', json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            # Log the full response for debugging
            return {"status": "sucess", "message": "Appointment Rescheduled Successfully", "response": result}
        else:
            return {"status": "error", "message": f"Received {response.status_code} from server."}

    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}


## Handle Tools Error
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls

    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    # Create a ToolNode with async error handling
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )