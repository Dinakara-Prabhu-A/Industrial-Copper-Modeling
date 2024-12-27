import streamlit as st
from streamlit_option_menu import option_menu
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import numpy as np

# Set up the page configuration
st.set_page_config(
    page_title="Industrial Copper Modeling",
    page_icon="📊",  # Choose a relevant icon
    layout="wide",  # You can use "centered" or "wide"
    initial_sidebar_state="expanded"  # Sidebar can be "auto", "expanded", or "collapsed"
)

st.markdown(
    f"""
    <div style="text-align: center;">
        <h1>Industrial Copper Modeling</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# Option Menu for selecting Regression or Classification
selected_task = option_menu(
    None, 
    ["Selling Price Prediction", "Status Prediction"], 
    icons=['graph-up', 'tags'], 
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal"
)

def reg_predict_datapoint(main_container):
    col1, col2 = main_container.columns(2, gap="medium")
    
    with col1:
        container = st.container(border = True)
        # Custom placeholder: Displaying an indicative label
        quantity_tons = container.number_input(
            "Quantity (Tons) (Enter a value between -2000 to 1000000000)", 
            min_value=-2000, 
            max_value=1000000000, 
            value=1,  # Default value as placeholder
            step=1, 
            key="quantity_tons"
        )
        
        container = st.container(border = True)
        country_list = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        country = container.selectbox(
            "Country Code", 
            country_list, 
            index=None, 
            placeholder="Select a country ",
            key="country_code"
        )
        
        container = st.container(border = True)
        product_ref = container.number_input(
            "Product Reference (Enter a value between 611728 to 1722207579)", 
            min_value=611728, 
            max_value=1722207579, 
            value=611728,  # Default value as placeholder
            step=1, 
            key="product_ref"
        )
        
        container = st.container(border = True)
        thickness = container.number_input(
            "Thickness (Enter a value between 0 to 2000)", 
            min_value=0, 
            max_value=2000, 
            value=0,  # Default value as placeholder
            step=1, 
            key="thickness"
        )
        
        container = st.container(border = True)
        width = container.number_input(
            "Width (Enter a value between 1 to 2990)", 
            min_value=1, 
            max_value=2990, 
            value=1,  # Default value as placeholder
            step=1, 
            key="width"
        )

    with col2:
        container = st.container(border = True)
        item_type_list = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        item_type = container.selectbox(
            "Item Type", 
            item_type_list, 
            index=None,
            placeholder="Select an item type", 
            key="item_type"
        )
        
        container = st.container(border = True)
        application_list = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        application = container.selectbox(
            "Application", 
            application_list, 
            index=None,
            placeholder="Select an application", 
            key="application"
        )
        
        container = st.container(border = True)
        item_date = container.date_input(
            "Item Date", 
            value=None, 
            key="item_date"
        )
        
        container = st.container(border = True)
        delivery_date = container.date_input(
            "Delivery Date", 
            value=None, 
            key="delivery_date"
        )
        
        container = st.container(border = True)
        status_list = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
        status = container.selectbox(
            "Status", 
            status_list, 
            index=None, 
            placeholder="Select a status",
            key="status"
        )
    
    if quantity_tons and country and product_ref and thickness and width and item_type and application and item_date and delivery_date and status:
        if st.button("CLICK  TO  PREDICT  THE PRICE",use_container_width=True):
            custom_data = CustomData(
                quantity_tons = quantity_tons,
                country = country,
                item_type = item_type,
                application = application,
                thickness = thickness,
                width = width,
                product_ref = product_ref,
                item_day = item_date.day,
                item_month = item_date.month,
                item_year = item_date.year,
                delivery_day = delivery_date.day,
                delivery_month = delivery_date.month,
                delivery_year = delivery_date.year,
                status = status
            )
            # Convert CustomData to DataFrame
            input_data = custom_data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            result = predict_pipeline.reg_predict(input_data)
            st.markdown(
    f"""
    <div style="text-align: center;">
        <h2>✨ Your Predicted Price Copper Product : 
        <span style="color: orange; font-weight: bold;">₹<strong> {np.expm1(result[0]):,.2f} </strong></span>
        ✨</h2>
        <hr style="border: 1px solid grey; width: 50%; margin: auto;">
        <p style="font-size:18px;">🏗️ Based on the inputs you provided, this is the estimated selling price for your copper product.</p>
    </div>
    """,
    unsafe_allow_html=True
)    

def class_predict_datapoint(main_container):
    col1, col2 = main_container.columns(2, gap="medium")
    
    with col1:
        container = st.container(border = True)
        # Custom placeholder: Displaying an indicative label
        quantity_tons = container.number_input(
            "Quantity (Tons) (Enter a value between -2000 to 1000000000)", 
            min_value=-2000, 
            max_value=1000000000, 
            value=1,  # Default value as placeholder
            step=1, 
            key="quantity_tons"
        )
        
        container = st.container(border = True)
        country_list = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        country = container.selectbox(
            "Country Code", 
            country_list, 
            index=None, 
            placeholder="Select a country ",
            key="country_code"
        )
        
        container = st.container(border = True)
        product_ref = container.number_input(
            "Product Reference (Enter a value between 611728 to 1722207579)", 
            min_value=611728, 
            max_value=1722207579, 
            value=611728,  # Default value as placeholder
            step=1, 
            key="product_ref"
        )
        
        container = st.container(border = True)
        thickness = container.number_input(
            "Thickness (Enter a value between 0 to 2000)", 
            min_value=0, 
            max_value=2000, 
            value=0,  # Default value as placeholder
            step=1, 
            key="thickness"
        )
        
        container = st.container(border = True)
        width = container.number_input(
            "Width (Enter a value between 1 to 2990)", 
            min_value=1, 
            max_value=2990, 
            value=1,  # Default value as placeholder
            step=1, 
            key="width"
        )

    with col2:
        container = st.container(border = True)
        item_type_list = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        item_type = container.selectbox(
            "Item Type", 
            item_type_list, 
            index=None,
            placeholder="Select an item type", 
            key="item_type"
        )
        
        container = st.container(border = True)
        application_list = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        application = container.selectbox(
            "Application", 
            application_list, 
            index=None,
            placeholder="Select an application", 
            key="application"
        )
        
        container = st.container(border = True)
        item_date = container.date_input(
            "Item Date", 
            value=None, 
            key="item_date"
        )
        
        container = st.container(border = True)
        delivery_date = container.date_input(
            "Delivery Date", 
            value=None, 
            key="delivery_date"
        )
        
        container = st.container(border = True)
        selling_price = container.number_input(
            "Selling Price (Enter a value between 1 to 1000000000)", 
            min_value=1, 
            max_value=100000000, 
            value=1,  # Default value as placeholder
            step=1, 
            key="selling_price"
        )
    
    if quantity_tons and country and product_ref and thickness and width and item_type and application and item_date and delivery_date and selling_price:
        if st.button("CLICK  TO  PREDICT  THE PRICE",use_container_width=True):
            custom_data = CustomData(
                quantity_tons = quantity_tons,
                country = country,
                item_type = item_type,
                application = application,
                thickness = thickness,
                width = width,
                product_ref = product_ref,
                item_day = item_date.day,
                item_month = item_date.month,
                item_year = item_date.year,
                delivery_day = delivery_date.day,
                delivery_month = delivery_date.month,
                delivery_year = delivery_date.year,
                selling_price = selling_price
            )
            # Convert CustomData to DataFrame
            input_data = custom_data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            result = predict_pipeline.class_predict(input_data)
            st.markdown(
    f"""
    <div style="text-align: center;">
        <h2>🚀 Lead Conversion Status</h2>
        <h3 style="color: {'green' if result == 0 else 'red'}; font-weight: bold;">
            {'Won - Congratulations!' if result == 0 else 'Lost - Better luck next time'}
        </h3>
        <hr style="border: 1px solid grey; width: 50%; margin: auto;">
        <p style="font-size:18px;">📈 Your lead has been analyzed, and this is the predicted status based on the data you provided.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Display corresponding task based on the selection
if selected_task == "Selling Price Prediction":
    main_container = st.container(border = True)
    reg_predict_datapoint(main_container)
    

elif selected_task == "Status Prediction":
    main_container = st.container(border = True)
    class_predict_datapoint(main_container)
    

                 
                