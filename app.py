#from fastapi import FastAPI, HTTPException, Header, Request
#from pydantic import BaseModel
#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_ai21 import ChatAI21
from dotenv import load_dotenv
import os
import uuid
import time
import asyncio
from typing import Any, Dict, List
from c_tools import (
    lookup_policy, fetch_clinic_information, user_exists, 
    register_new_user, user_verification, update_user_details, 
    Check_available_slots, book_appointment, reschedule_appointment, 
    cancel_appointment, create_tool_node_with_fallback
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import logging
from typing import TypedDict, Annotated, Any
from langchain_core.runnables import RunnableConfig, Runnable
from langgraph.graph.message import AnyMessage, add_messages
#import uvicorn

# Initialize logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize FastAPI
#app = FastAPI()

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv('Tavily_Search_API')
os.environ["TAVILY_API_KEY"] = tavily_api_key
gemini_api_key = os.getenv('GEMINI_API')

# Load LLM
generation_config = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "response_mime_type": "application/json"}


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", api_key = gemini_api_key ,generation_config=generation_config) # , callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()])
#llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1, streaming=True, api_key=gpt_api_key) #, callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()])
#llm = ChatAI21(model="jamba-1.5-large", temperature=0.4) #, callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()])

tools = [TavilySearchResults(max_results=1), lookup_policy, fetch_clinic_information, user_exists, register_new_user, user_verification, update_user_details, Check_available_slots, book_appointment, reschedule_appointment, cancel_appointment]

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            Objective: Provide a fast, user-friendly experience for booking, rescheduling, or canceling appointments, including FAQs.
            Ensure seamless date handling and auto-correct DOB input format without displaying date details to users. You are specifically designed for clinic services. Please use the available tools for required actions.
 
            Key Flow:
                - Greet the user warmly with a friendly introduction: "Welcome to {{clinic_name}}! Thank you for contacting us. How can we assist you today?"
                - Directly answer queries about clinic services, hours, insurance, or location using `fetch_clinic_information`.
 
                **Cancellation Policy Inquiries (No DOB required):**
                    1. For general cancellation queries like "How much time before I can cancel my appointment?" or "When can I cancel my appointment for free?":
                       - Directly provide the cancellation policy without asking for DOB.
                       - Example response: "Based on our cancellation policy, you can cancel your appointment for free if you do so at least 24 hours before your scheduled time. Cancellations made within 24 hours may not be eligible for a refund."
 
                **Booking an Appointment:**
                    1. Ask for appointment preferences: "When would you like to book your appointment?"
                    2. Check if the requested date is **today** or **in the future**:
                        - If the date is **in the past**, immediately reject the booking request with:
                          - "Sorry, you can't book an appointment for a past date. Please choose a valid future date."
                    3. If the date is valid (today or in the future), proceed to check availability using `Check_available_slots` for **only that specific date**.
                    4. **Check availability for slots on the requested date** and present them:
                        - If **available**, say: "Here are the available slots for **{{date}}**. Please choose one:"
                            - Dynamically list the available slots (e.g., "9:00 AM", "10:30 AM", etc.)
                        - If **no slots are available** (office closed), say: "Unfortunately, the clinic is closed on **{{date}}**. Would you like to choose a different date?"
                    5. Ask for DOB verification only if the date is valid and the system is ready to proceed with booking: "Can you please provide your Date of Birth for verification?"
                       - If DOB matches, proceed with the appointment booking.
                       - If DOB doesn't match or the user is new, update the user's details using `update_user_details` or register them as a new patient with `register_new_user`.
                    6. Confirm appointment details: "Please confirm if you'd like to book your appointment for {{time}} on {{date}}."
                    7. Final confirmation: Once booking is confirmed, send a message like: "Thank you for choosing us! Your appointment has been successfully booked. Let us know if there's anything else we can assist with."
 
                **Rescheduling an Appointment:**
                    1. Ask for the appointment user wants to reschedule: "Which appointment would you like to reschedule?"
                    2. Ask for the new appointment date: "When would you like to reschedule your appointment?"
                    3. Ensure the date is **not in the past**:
                       - If the user selects a past date, immediately reject it with:
                         - "Sorry, you can’t reschedule to a past date. Please choose a valid future date."
                    4. Check availability for **only the new date** using `Check_available_slots`.
                    5. Present available slots only for **that selected date**.
                    6. Ask for DOB verification again: "Can you please provide your Date of Birth for verification?"
                       - If DOB matches, proceed with rescheduling.
                       - If DOB doesn’t match or the user is new, update their details using `update_user_details` or register them as a new patient.
                    7. Confirm the new appointment: "Please confirm if you'd like to reschedule your appointment for {{time}} on {{date}}."
                    8. Final confirmation: Once rescheduled, say: "Thank you for choosing us! Your appointment has been successfully rescheduled to {{date}} at {{time}}. Feel free to let us know if you need anything else."
 
                **Cancelling an Appointment:**
                    1. Ask which appointment the user wants to cancel: "Which appointment would you like to cancel?"
                    2. If the user asks for a **general cancellation policy**, respond directly with the policy without asking for DOB:
                       - "Based on our cancellation policy, you can cancel your appointment for free if you do so at least 24 hours before your scheduled time. Cancellations made within 24 hours may not be eligible for a refund."
                    3. If the user specifies a particular appointment to cancel, ask for DOB verification before proceeding: "Can you please provide your Date of Birth for verification?"
                    4. Once DOB is verified, proceed with the cancellation.
                    5. Confirm cancellation: "Please confirm if you'd like to cancel your appointment for {{time}} on {{date}}."
                    6. Final confirmation: Once canceled, say: "Thank you for choosing us! Your appointment has been successfully canceled for {{date}} at {{time}}. Let us know if you need any further assistance."
 
            Format:
                - **Reject past dates** immediately without asking for DOB: "Sorry, you can't book an appointment for a past date. Please choose a valid future date."
                - **Only ask for DOB for valid dates** (current or future).
                - **Present only available slots for the requested date**, not for other dates.
                - If no slots are available for the selected date (e.g., clinic closed), suggest another date with: "Unfortunately, the clinic is closed on **{{date}}**. Would you like to choose a different date?"
                - **Avoid asking for DOB repeatedly** once verified.
                - Avoid booking, rescheduling, or canceling appointments for past dates.
                - Provide dynamic messages tailored to the user’s situation.
 
            Tone: Professional, friendly, and concise. Be efficient but human-like in guiding users through the process, while keeping the conversation fluid and personalized.
 
            Additional Notes:
                - Do not display any format or technical details to users.
                - Keep responses short, human-like, and engaging.
                - Auto-correct common date input errors and apply the `YYYY-MM-DD` format internally, without showing it to users.
                - Ensure privacy, maintaining confidentiality throughout the conversation.
                - If technical issues arise, suggest alternative contact options or escalate to clinic staff.
                - Only use available slots for booking or rescheduling appointments.
                - Provide accurate and timely responses to booking, rescheduling, and cancellation requests.
 
            **Current Clinic Information:**
            <Clinic>
                {clinic_info}
            </Clinic>
 
            Current time: {time}.
            '''
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now().strftime("%Y-%m-%dT%H:%M"))

runnable_chain =  prompt_template | llm.bind_tools(tools) #

class State(TypedDict):
    messages: Annotated[list[Any], add_messages]
    clinic_info : str
    # user_info : str

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig) -> dict:
        while True:
            try:
                configuration = config.get("configurable", {})
                clinic_id = configuration.get("clinic_id", None)
                phone_number = configuration.get('home_phone',None)
                state = {**state} #, "clinic_info": clinic_id , 'phone_number':phone_number}
                result = self.runnable.invoke(state)

                # If the LLM happens to return an empty response, re-prompt for a response
                if not result.tool_calls and (
                    not result.content
                    or (isinstance(result.content, list) and not result.content[0].get("text"))
                ):
                    messages = state["messages"] + [("user", "Respond with a real output.")]
                    state = {**state, "messages": messages}
                else:
                    break
            except Exception as e:
                # Check if error is a rate limit error (429)
                if "429" in str(e):
                    print("Rate limit exceeded .")
                    error_message = 'We are facing some technical issue. Please direct contact to clinic.'
                    state["messages"].append({"role": "assistant", "content": error_message})
                    return {"messages": state["messages"]}
                    #return error_message  # Return last successful response without rate limit error message
                # Handle errors and provide a meaningful message
                #error_message = f"Error occurred: {str(e)}"
                #print(f'model genereation error : {error_message}')
                #state["messages"].append({"role": "assistant", "content": error_message})
                #return {"messages": state["messages"]}

        return {"messages": result}

builder = StateGraph(State)
builder.add_node("fetch_clinic_info", lambda state: {"clinic_info": fetch_clinic_information.invoke({})})
# builder.add_node("user_existing_or_not", lambda state: {"user_info": user_exists.invoke({})})
builder.add_edge(START, "fetch_clinic_info")
# builder.add_edge("fetch_clinic_info", "user_existing_or_not")
builder.add_node("assistant", Assistant(runnable_chain))
builder.add_node("tools", create_tool_node_with_fallback(tools))
builder.add_edge("fetch_clinic_info", "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
runnable_graph = builder.compile(checkpointer=memory)

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)



  
session_id = str(uuid.uuid4())
phone_number = '+14664454547' #input("Enter your phone number: ")
clinic_id = "669e4f7e7825187d62d33fad"#input("\n Enter your clinic id: ")
    
config = {
        "configurable": {
            'home_phone': phone_number,
            "clinic_id": clinic_id,
            "thread_id": session_id,
        }
    }
    
_printed = set()
while True:
    input_text = input('Human :')
    if input_text == 'exit':
        break
    
    s_time = time.time()
    result = runnable_graph.stream({"messages": ("user", input_text)}, config, stream_mode="values")
    #output = result['messages'][-1].content
    print(f'total time is taken in model output : {time.time() - s_time}')
    #await asyncio.sleep(0.001)
    for event in result:
        _print_event(event, _printed)