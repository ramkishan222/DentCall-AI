from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import uuid
import asyncio
#import hashlib
from typing import Any, Dict, List
from c_tools import (
    lookup_policy, fetch_clinic_information,
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
#import redis
from typing import TypedDict, Annotated, Any
from langchain_core.runnables import RunnableConfig, Runnable
from langgraph.graph.message import AnyMessage, add_messages
import uvicorn

# Initialize logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Set up Redis connection
#redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Initialize FastAPI
app = FastAPI()

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv('Tavily_Search_API')
os.environ["TAVILY_API_KEY"] = tavily_api_key
#gpt_api_key = os.getenv('GPT_KEY')
gemini_api_key = os.getenv('GEMINI_API')
#os.environ["AI21_API_KEY"] = os.getenv('AI21_API_KEY')

# Load LLM
# generation_config = {
#             "temperature": 1,
#             "top_p": 1,
#             "top_k": 1,
#             "response_mime_type": "application/json"}

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 1,
  #"top_k": 40,
  "max_output_tokens": 200,
  "response_mime_type": "application/json",
}


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", api_key = gemini_api_key ,generation_config=generation_config) # , callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()])
#llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.4, streaming=True, api_key=gpt_api_key) #, callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()])
#llm = ChatAI21(model="jamba-1.5-mini", temperature=0.4) #, callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()])

tools = [
        TavilySearchResults(max_results=1), 
        lookup_policy, fetch_clinic_information, register_new_user, user_verification, 
        update_user_details, Check_available_slots, book_appointment, reschedule_appointment, cancel_appointment]

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            Objective: Provide a fast and user-friendly experience for booking, rescheduling, or canceling appointments, including FAQs.
                       Ensure seamless date handling and auto-correct DOB input format without displaying date details to users.
                       You are specially design for clinic only.and use these tools accordings to requirements auto.
                       Any clinic related information provide them by using clinic_info.

            Key Flow:
                - Start with a friendly greeting with mentioning as 'Welcome to {{clinic_name}} . Thank you for contacting us . How can we help for you.'
                - Respond directly to clinic service, hours, insurance, or location inquiries. Use `fetch_clinic_information`.
                
                - For book appointment requests, confirm actions without showing specific dates or DOB format.:
                    1. Request for Appointment like 'When would you like to book your appointment.'
                    2. Check Slots is available or not that day use `Check_available_slots`,  as needed..
                    3. Provide available slots like 'Here available slots are [slots]. you can choice which slot you want to book for your appointment'.
                    4. Just before final booking or rescheduling or cancling appointment Request DOB in a user-friendly manner like 'Can you please provide me your Date of Birth for verification purpose' for `user_verification`, auto-correct format as needed.
                       - If User Exist and DOB sucessfully matched then book appointment.
                       - If User Exist and DOB not matched then update user details using `update_user_details' to capture required details.
                       - If DOB is not matched and User Does not Exist then register it as new patients, use `register_new_user` to capture required details.
                    5. When You successfully verify that user is exist and verified or registerd as new user.
                    6. Then Final Booking Process:
                        - Then Request a conformation for appointment booking as 'Please Can you conform to book your appointment for [time] on [date]'.
                        - When appointment is successfully book by using `book_appointment` then provide a message like 'Thank you for choising us. Your appointment have sucessfully Booked. If there any please let me know'
                        
                - For rescheduling appointment requests, confirm actions without showing specific dates or DOB format.:
                    1. Request for old appointment on which user want to reschedule like 'Which appointment would you like to reschedule.'
                    2. Check Cancle policy using `lookup_policy`
                        - If cancle policy does not match then response like 'I can't reschedule your appointment before 24 hours . If you want to processed this please contact to clinic direct'.
                    3. Request for rescheduling appointment like 'When would you like to reschedule your appointment.'
                    4. Check Slots is available or not that day use `Check_available_slots`,  as needed..
                    5. Provide available slots like 'Here available slots are {{slots}}. you can choice which slot you want to book for your new appointment'.
                    6. Just before final rescheduling appointment Request DOB in a user-friendly manner like 'Can you please provide me your Date of Birth for verification purpose' for `user_verification`, auto-correct format as needed.
                       - If User Exist and DOB sucessfully matched then reschedule appointment.
                       - If User Exist and DOB not matched then update user details using `update_user_details' to capture required details.
                       - If DOB is not matched and User Does not Exist then register it as new patients, use `register_new_user` to capture required details.
                    7. When You successfully verift that user is exist and verified or registerd as new user.
                    8. Final rescheduling Process:
                        - Then Request a conformation for appointment rescheduling as 'Please Can you conform to reschedule your appointment for [time] on [date]'.
                        - When appointment successfully reschedule by using `reschedule_appointment` for rescheduling appointemnt with capture required details.
                        - then provide a message like 'Thank you for choising us. Your appointment have sucessfully rescheduled on [date] [time]. If there any please let me know'
                            
                - For cancellation appointment requests, confirm actions without showing specific dates or DOB format.:
                    1. Request for old appointment on which user want to cancle like 'Which appointment would you like to cancle.'
                    2. Check Cancle policy using `lookup_policy`
                        - If cancle policy does not match then response like 'I can't cancle your appointment before 24 hours . If you want to processed this please contact to clinic direct'.
                    3. Just before final cancellation appointment Request DOB in a user-friendly manner like 'Can you please provide me your Date of Birth for verification purpose' for `user_verification`, auto-correct format as needed.
                       - If User Exist and DOB sucessfully matched then cancling appointment.
                       - If User Exist and DOB not matched then update user details using `update_user_details' to capture required details.
                       - If DOB is not matched and User Does not Exist then register it as new patients, use `register_new_user` to capture required details.
                    7. When You successfully verift that user is exist and verified or registerd as new user.
                    8. Final cancellation Process:
                        - Then Request a conformation for appointment cancellation as 'Please Can you conform to cancle your appointment for [time] on [date]'.
                        - When appointment successfully cancle by using `cancel_appointment` for cancle appointemnt with capture required details.
                        - Then provide a message like 'Thank you for choising us. Your appointment have sucessfully cancled on [date] [time]. If there any please let me know'

            Format:
                - Greet the user once at the start and ask about their needs.
                - Verify patient status for appointment-related requests.
                - Guide through booking, rescheduling, or cancellation without displaying any format requirements.
                - Confirm details clearly but avoid mentioning specific date or time formats.
                - Keep patient details confidential and avoid discussing medical specifics.
                - Alway use enum value for gender : ["Male", "Female", "Other", "Unknown"] only.
                

            Tone: Professional, friendly, and concise. Be direct and efficient in guiding users through the process . Avoid to provide outside information.

            Additional Notes:
                - Do not disply any format to users.
                - Provide response in one sentensce only.
                - Response will be proper human level only.
                - If there any out of knowledge then "we do not serve this here" 
                - Auto-correct common date input errors and apply `YYYY-MM-DD` format without displaying this to users.
                - Ensure patient privacy, maintaining confidentiality throughout the conversation.
                - If technical issues arise, suggest alternative contact options.
                - If user wants to cancle or reschedule appointment then check cancle policy using `lookup_policy` first.
                - If unable to answer, offer to connect with clinic staff for further assistance.
                
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
                # configuration = config.get("configurable", {})
                # clinic_id = configuration.get("clinic_id", None)
                # phone_number = configuration.get('home_phone',None)
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

async def chatbot_session(input_text, phone_number, clinic_id, thread_id):
    config = {
        "configurable": {
            'home_phone': phone_number,
            "clinic_id": clinic_id,
            "thread_id": thread_id,
        }
    }
    #result = runnable_graph.batch([{"messages": ("user", input_text)}], config, stream_mode="values")
    result = runnable_graph.invoke({"messages": [("user", input_text)]}, config, stream_mode="values")
    await asyncio.sleep(0.001)
    return result['messages'][-1].content #result[-1]['messages'][-1].content

class ChatRequest(BaseModel):
    human: str

@app.get("/")
def home():
    return {"message": "Welcome to the Chatbot AI by LAKHAN SINGH!"}

@app.post("/api/chatbot/clinic_ai")
async def chatbot(request: Request, chat_request: ChatRequest, x_api_key: str = Header(None)):
    if x_api_key != os.getenv('X-API-KEY'):
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Query parameters
    clinic_id = request.query_params.get('clinic_id')
    patient_number = request.query_params.get('patient_number')
    session_id = request.query_params.get('session_id')

    if not clinic_id or not patient_number or not session_id:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    formatted_phone_number = patient_number.strip().replace(" ", "")
    if not formatted_phone_number.startswith("+"):
        formatted_phone_number = f"+{formatted_phone_number}"

    user_input = chat_request.human
    if not user_input:
        raise HTTPException(status_code=400, detail="Please provide input text")

    try:
        chatbot_response = await chatbot_session(user_input, formatted_phone_number, clinic_id, session_id)
        return {"model_response": chatbot_response}
    
    except Exception as e:
        logging.error(f"Error during chatbot session: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during chatbot session")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)