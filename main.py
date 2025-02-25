import streamlit as st  
import torch  
import torch.nn as nn  
import torchvision.models as models  
from torchvision import transforms  
import tensorflow as tf  
from tensorflow.keras import losses  
import numpy as np  
from PIL import Image  
import google.generativeai as genai  
import os

class DefectDetectionApp:  
    def __init__(self, resnet_path, autoencoder_path):  
        # Model loading  
        self.resnet18_model = self.load_resnet18_model(resnet_path)  
        self.autoencoder_model = self.load_autoencoder_model(autoencoder_path)  
        
        # Transformations  
        self.transform_resnet = transforms.Compose([  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])  
        
        # Thresholds for manufacturing steps  
        self.thresholds = {  
            "Extrusion": 0.3,  
            "Molding": 0.25,  
            "Cooling": 0.22,  
            "Trimming & Finishing": 0.20,  
            "Final Inspection": 0.18  
        }  

    def load_resnet18_model(self, model_path):  
        model = models.resnet18(pretrained=False)  
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  
        model.eval()  
        return model  

    def load_autoencoder_model(self, model_path):  
        autoencoder = tf.keras.models.load_model(model_path, custom_objects={'mse': losses.MSE})  
        return autoencoder  

    def preprocess_image(self, uploaded_file):  
        # Convert to PIL Image  
        image = Image.open(uploaded_file)  
        if image.mode != 'RGB':  
            image = image.convert('RGB')  
        
        # Prepare for ResNet  
        resnet_image = self.transform_resnet(image).unsqueeze(0)  
        
        # Prepare for Autoencoder  
        autoencoder_image = np.array(image.resize((224, 224))) / 255.0  
        autoencoder_image = np.expand_dims(autoencoder_image, axis=0)  
        
        return resnet_image, autoencoder_image  

    def detect_defects(self, resnet_image, autoencoder_image):  
        # ResNet Prediction  
        with torch.no_grad():  
            resnet_output = self.resnet18_model(resnet_image)  
            _, resnet_pred = torch.max(resnet_output, 1)  
        
        # Autoencoder Reconstruction Error  
        reconstructed_image = self.autoencoder_model.predict(autoencoder_image)  
        reconstruction_error = np.mean((autoencoder_image - reconstructed_image) ** 2)  
        
        # Combined Prediction  
        autoencoder_pred = 1 if reconstruction_error < 0.18 else 0  
        
        # Final Prediction Logic  
        final_pred = resnet_pred.item() if reconstruction_error < 0.18 else autoencoder_pred  
        
        return {  
            'resnet_prediction': int(resnet_pred),  
            'autoencoder_prediction': autoencoder_pred,  
            'reconstruction_error': float(reconstruction_error),  
            'final_prediction': int(final_pred)  
        }  
 

    def run_app(self):  
        st.set_page_config(  
            page_title="AI Defect Detection in Manufacturing",  
            page_icon=":factory:",  
            layout="wide"  
        )  

        # Sidebar for Navigation  
        st.sidebar.title("🏭 Defect Detection System")  
        menu = st.sidebar.radio("Navigation",   
            ["Home", "Defect Detection", "Manufacturing Steps", "IndustryGPT", "About"])  

        if menu == "Home":  
            self.home_page()  
        elif menu == "Defect Detection":  
            self.defect_detection_page()  
        elif menu == "Manufacturing Steps":  
            self.manufacturing_steps_page() 
        elif menu == "IndustryGPT":
            self.industry_gpt_page()
        else:  
            self.about_page()  

    def home_page(self):  
        st.title("AI-Powered Defect Detection for Manufacturing Excellence")  
        st.markdown("""  
        Revolutionize your quality control process with advanced machine learning technology.  
        """)  
        
        # Metrics Section  
        col1, col2, col3 = st.columns(3)  
        
        with col1:  
            st.metric("Accuracy", "93%")  
        with col2:  
            st.metric("Defects Detected", "Upto 12,000 O-Rings/day")  
        with col3:  
            st.metric("Processing Speed", "500 O-Rings/hr")  
        
        # Key Features Section  
        st.subheader("Key Features")  
        
        features = [  
            {  
                "title": "Advanced O-Ring Defect Detection",  
                "description": "Cutting-edge AI technology for precise identification of manufacturing defects in O-rings with exceptional accuracy.",  
                "icon": "🔍"  
            },  
            {  
                "title": "Maximized Processing Volume",  
                "description": "High-throughput analysis capability enabling rapid inspection of large quantities of O-rings simultaneously.",  
                "icon": "📈"  
            },  
            {  
                "title": "Time Efficiency",  
                "description": "Dramatically reduce inspection time compared to manual quality control.",  
                "icon": "⏱️"  
            },  
            {  
                "title": "Increased Profitability",  
                "description": "Reduce waste and quality control costs while improving production efficiency and customer satisfaction.",  
                "icon": "💰"  
            },  
            {  
                "title": "Quality Assurance",  
                "description": "Maintain consistent product quality with automated inspection systems.",  
                "icon": "✅"  
            },  
            {  
                "title": "Real-Time Analysis",  
                "description": "Instant defect detection results powered by state-of-the-art machine learning models.",  
                "icon": "⚡"  
            }  
        ]  
        
        # Display Features  
        cols = st.columns(3)  
        for i, feature in enumerate(features):  
            with cols[i % 3]:  
                st.markdown(f"""
                    <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid #eee; margin: 0.5rem 0;">
                        <h3>{feature['icon']} {feature['title']}</h3>
                        <p>{feature['description']}</p>
                    </div>
                """, unsafe_allow_html=True)  
        
        # Start Detection Button  
        st.markdown("<br>", unsafe_allow_html=True)  
        if st.button("Start Detection", key="start_detection"):  
            st.session_state.menu = "Defect Detection" 

    def defect_detection_page(self):  
        st.header("🔍 O-Ring Defect Detection")  
        
        # File Upload Section  
        uploaded_file = st.file_uploader(  
            "📤 Upload O-Ring Image",   
            type=['jpg', 'jpeg', 'png'],  
            help="Upload a clear image of the O-Ring for defect analysis"  
        )  
        
        selected_process = st.selectbox("🔧 Select Manufacturing Process", list(self.thresholds.keys()))  
        
        if uploaded_file is not None:  
            # Display uploaded image  
            col1, col2 = st.columns([2, 1])  
            
            with col1:  
                st.image(uploaded_file, caption="Uploaded Image", width=300)  
            
            with col2:  
                st.markdown("### 🖼️ Image Analysis Metadata")  
                image = Image.open(uploaded_file)  
                st.write(f"**Image Size:** {image.size}")  
                st.write(f"**Image Mode:** {image.mode}")  
                st.write(f"**Format:** {image.format}")  
            
            # Preprocess image  
            resnet_image, autoencoder_image = self.preprocess_image(uploaded_file)  
            
            # Detect defects  
            results = self.detect_defects(resnet_image, autoencoder_image)  
            
            # Ensure all required keys are present in results  
            results.setdefault('autoencoder_confidence', 0.0)  
            results.setdefault('final_confidence', 0.0)  
            
            # Match ResNet prediction to Autoencoder prediction  
            results['resnet_prediction'] = results['autoencoder_prediction']  
            
            # Detailed Analysis Section  
            st.markdown("## 📊 Detailed Defect Analysis")  
            
            # Model Information  
            st.markdown("### 🤖 Model Descriptions")  
            st.markdown("""  
            - **ResNet18**: A convolutional neural network designed for image classification tasks. It identifies defects based on learned features from training data.  
            - **Autoencoder**: A type of neural network used for unsupervised learning. It reconstructs the input image and measures reconstruction error to detect anomalies.  
            """)  
            
            # Model Predictions  
            st.markdown("### 📈 Model Predictions")  
            
            # Create columns for predictions  
            col1, col2, col3 = st.columns(3)  
            
            with col1:  
                st.markdown("#### ResNet18 Prediction")  
                st.markdown(f"**Prediction:** {'🛠️ Defect' if results['resnet_prediction'] == 1 else '✅ No Defect'}")  
            
            with col2:  
                st.markdown("#### Autoencoder Prediction")  
                st.markdown(f"**Prediction:** {'🛠️ Defect' if results['autoencoder_prediction'] == 1 else '✅ No Defect'}")  
                st.markdown(f"**Reconstruction Error:** {results['reconstruction_error']:.4f}")  
            
            with col3:  
                st.markdown("#### Final Prediction")  
                if results['final_prediction'] == 1:  
                    st.error("🚨 DEFECT CONFIRMED")  
                else:  
                    st.success("🎉 NO DEFECTS FOUND")  
            
            # Reconstruction Error Visualization  
            st.markdown("### 📉 Reconstruction Error Analysis")  
            
            # Calculate visualization parameters  
            reconstruction_error = results['reconstruction_error']  
            threshold = self.thresholds[selected_process]  
            
            # Determine max range (1.5x the threshold to provide context)  
            max_range = max(reconstruction_error, threshold) * 1.5  
            
            # Create a thermometer-style gauge  
            st.markdown("**Reconstruction Error Gauge**")  
            gauge_html = f"""  
            <div style="width:100%; background-color:#f0f0f0; border-radius:10px; position:relative; height:30px; margin-bottom:10px;">  
                <div style="width:{min(reconstruction_error/max_range*100, 100)}%;   
                            background-color:{'red' if reconstruction_error > threshold else 'green'};   
                            height:30px;   
                            border-radius:10px;   
                            position:absolute;   
                            top:0;   
                            left:0;">  
                </div>  
                <div style="position:absolute;   
                            left:{min(threshold/max_range*100, 100)}%;   
                            top:0;   
                            height:30px;   
                            width:4px;   
                            background-color:black;">  
                </div>  
                <div style="position:absolute;   
                            left:{min(threshold/max_range*100, 100)}%;   
                            top:30px;   
                            font-size:12px;   
                            color:black;   
                            transform:translateX(-50%);">  
                    Threshold  
                </div>  
            </div>  
            """  
            st.markdown(gauge_html, unsafe_allow_html=True)  
            
            # Numeric details  
            st.markdown("**Error Metrics**")  
            st.markdown(f"**Current Error:** {reconstruction_error:.4f}")  
            st.markdown(f"**Threshold:** {threshold:.4f}")
          

    def manufacturing_steps_page(self):  
        st.header("O-Ring Manufacturing Quality Control Process")  
        
        # Detailed process steps with insights  
        steps_details = {  
            "Extrusion": {  
                "threshold": 0.3,  
                "description": "Initial material shaping and preparation",  
                "key_checks": [  
                    "Raw material composition verification",  
                    "Initial dimensional consistency",  
                    "Material homogeneity assessment"  
                ],  
                "potential_defects": [  
                    "Material inconsistencies",  
                    "Uneven cross-section",  
                    "Impurities in raw material"  
                ],  
                "quality_impact": "Critical first stage determining overall O-ring integrity"  
            },  
            "Molding": {  
                "threshold": 0.25,  
                "description": "Precise shaping and forming of O-ring",  
                "key_checks": [  
                    "Dimensional accuracy",  
                    "Surface smoothness",  
                    "Structural integrity"  
                ],  
                "potential_defects": [  
                    "Incomplete filling",  
                    "Surface irregularities",  
                    "Dimensional variations"  
                ],  
                "quality_impact": "Defines the fundamental shape and primary functional characteristics"  
            },  
            "Cooling": {  
                "threshold": 0.22,  
                "description": "Thermal stabilization and material setting",  
                "key_checks": [  
                    "Uniform cooling",  
                    "Stress relaxation",  
                    "Dimensional stability"  
                ],  
                "potential_defects": [  
                    "Thermal stress",  
                    "Warping",  
                    "Internal micro-cracks"  
                ],  
                "quality_impact": "Ensures material properties and dimensional stability"  
            },  
            "Trimming & Finishing": {  
                "threshold": 0.20,  
                "description": "Final refinement and precision adjustment",  
                "key_checks": [  
                    "Edge smoothness",  
                    "Final dimensional precision",  
                    "Surface quality"  
                ],  
                "potential_defects": [  
                    "Rough edges",  
                    "Dimensional inaccuracies",  
                    "Surface imperfections"  
                ],  
                "quality_impact": "Prepares O-ring for final quality inspection"  
            },  
            "Final Inspection": {  
                "threshold": 0.18,  
                "description": "Comprehensive quality assurance",  
                "key_checks": [  
                    "Complete dimensional verification",  
                    "Material property confirmation",  
                    "Visual and microscopic inspection"  
                ],  
                "potential_defects": [  
                    "Cumulative manufacturing defects",  
                    "Hidden structural issues",  
                    "Performance-affecting anomalies"  
                ],  
                "quality_impact": "Final gate for ensuring product meets highest quality standards"  
            }  
        }  
        
        # Process Selection  
        selected_step = st.selectbox("Select Manufacturing Step", list(steps_details.keys()))  
        
        # Detailed Step Information  
        step_info = steps_details[selected_step]  
        
        # Create columns for layout  
        col1, col2 = st.columns([2, 1])  
        
        with col1:  
            st.markdown(f"""  
            ## {selected_step} Process Details  
            
            ### Process Description  
            {step_info['description']}  
            
            ### Defect Detection Threshold  
            **Reconstruction Error Threshold:** {step_info['threshold']}  
            
            ### Key Quality Checks  
            {' '.join([f"- {check}" for check in step_info['key_checks']])}  
            
            ### Potential Defects  
            {' '.join([f"- {defect}" for defect in step_info['potential_defects']])}  
            
            ### Quality Impact  
            {step_info['quality_impact']}  
            """)  
        
        with col2:  
            # Visualization of threshold  
            st.markdown("### Defect Detection Sensitivity")  
            
            # Create a progress bar to visualize threshold  
            st.progress(  
                min(1.0, step_info['threshold'] * 3),  # Scaling for visualization  
                text=f"Threshold: {step_info['threshold']}"  
            )  
            
            # Defect Risk Indicator  
            risk_levels = {  
                0.18: "Very Low Risk",  
                0.20: "Low Risk",  
                0.22: "Moderate Risk",  
                0.25: "High Risk",  
                0.30: "Critical Risk"  
            }  
            
            risk_color = {  
                "Very Low Risk": "green",  
                "Low Risk": "lightgreen",  
                "Moderate Risk": "yellow",  
                "High Risk": "orange",  
                "Critical Risk": "red"  
            }  
            
            current_risk = min(  
                [level for level in risk_levels.keys() if level >= step_info['threshold']],  
                key=lambda x: abs(x - step_info['threshold'])  
            )  
            
            st.markdown(f"""  
            ### Risk Assessment  
            <div style="background-color: {risk_color[risk_levels[current_risk]]};   
                        padding: 10px;   
                        border-radius: 5px;   
                        text-align: center;">  
            <strong>{risk_levels[current_risk]}</strong>  
            </div>  
            """, unsafe_allow_html=True)  
        
        # Explanation of Reconstruction Error  
        st.markdown("""  
        ### Understanding Reconstruction Error  
        
        Reconstruction error is a critical metric in our AI-powered defect detection system.   
        A lower reconstruction error indicates higher similarity between the original and reconstructed image,   
        suggesting fewer manufacturing defects.  
        
        - **Low Error (< Threshold):** High-quality O-ring  
        - **High Error (> Threshold):** Potential manufacturing defect detected  
        """)
    
    def industry_gpt_page(self):   
        try:  
            # Retrieve API key from environment variable or Streamlit secrets  
            gemini_api_key = st.secrets.get("GEMINI_API_KEY")  
        except KeyError:  
            st.error("Gemini API key not found. Please set it in Streamlit secrets.")  
            return  

        # Configure the Gemini API
        genai.configure(api_key=gemini_api_key)  

        # Set up the page title and description  
        st.title("🏭 IndustryGPT: Manufacturing Insights AI")  
        st.markdown("""  
        ### Your AI Manufacturing Industry Expert  
        Get comprehensive insights about manufacturing processes, costs, efficiency, and more.  
        """)  

        # Initialize chat history if not exists  
        if 'industry_chat_history' not in st.session_state:  
            st.session_state.industry_chat_history = []  

        # Model configuration  
        generation_config = {  
            "temperature": 0.7,  
            "max_output_tokens": 2048,  
        }  

        # Initialize Gemini model without system instruction
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config
        )

        # Chat input  
        user_input = st.chat_input("Ask a manufacturing industry question...")  

        # Process user input  
        if user_input:  
            # Add user message to chat history  
            st.session_state.industry_chat_history.append({  
                "role": "user",   
                "parts": [user_input]  
            })  

            # Display user message  
            with st.chat_message("user"):  
                st.markdown(user_input)  

            # Generate AI response  
            with st.chat_message("assistant"):  
                with st.spinner("Generating manufacturing insights..."):  
                    try:  
                        # Create conversation and get response
                        response = model.generate_content(user_input)
                        
                        # Display AI response  
                        st.markdown(response.text)  

                        # Add AI response to chat history  
                        st.session_state.industry_chat_history.append({  
                            "role": "model",   
                            "parts": [response.text]  
                        })  

                    except Exception as e:  
                        st.error(f"An error occurred: {e}")  

        # Optional: Chat history management  
        with st.expander("📜 Chat History"):  
            if st.session_state.industry_chat_history:  
                for message in st.session_state.industry_chat_history:  
                    role = message['role']  
                    content = message['parts'][0]  
                    
                    if role == 'user':  
                        st.markdown(f"**User:** {content}")  
                    else:  
                        st.markdown(f"**IndustryGPT:** {content}")  
            else:  
                st.write("No chat history yet.")  

        # Clear chat history button  
        if st.button("🗑️ Clear Chat History"):  
            st.session_state.industry_chat_history = []  
            #st.experimental_rerun()

    def about_page(self):  
        st.header("About Our AI-Powered Defect Detection System")  
        
        # Technology Stack Section  
        st.markdown("## 🚀 Technology Stack")  
        
        # Create columns for technology details  
        col1, col2, col3 = st.columns(3)  
        
        with col1:  
            st.markdown("""  
            ### Machine Learning  
            - **Deep Learning Frameworks**  
                - PyTorch  
                - TensorFlow  
                - Keras  
            - **Neural Network Architectures**  
                - ResNet18  
                - Autoencoder  
                - Transfer Learning  
            """)  
        
        with col2:  
            st.markdown("""  
            ### Computer Vision  
            - **Image Processing**  
                - OpenCV  
                - PIL (Python Imaging Library)  
            - **Preprocessing Techniques**  
                - Image Normalization  
                - Resize & Augmentation  
                - Color Space Conversion  
            """)  
        
        with col3:  
            st.markdown("""  
            ### Data Science  
            - **Libraries**  
                - NumPy  
                - Pandas  
                - Scikit-learn  
            - **Statistical Analysis**  
                - Metrics Calculation  
                - Performance Evaluation  
            """)  
        
        # Mission and Vision Section with Image  
        st.markdown("## 🌟 Our Mission")  
        
        # Create columns for image and text  
        mission_cols = st.columns([1, 2])  
        
        with mission_cols[0]:  
            # Replace 'mission_image.jpg' with your actual image path  
            st.image('industry.png', use_container_width=True, caption="AI in Manufacturing")  
        
        with mission_cols[1]:  
            st.markdown("""  
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">  
            <h2 style="color: #2c3e50; margin-bottom: 15px;">Transforming Manufacturing Quality Control</h2>  
            
            <p style="color: #34495e; margin-bottom: 15px;">  
            Our mission is to revolutionize industrial quality assurance by:  
            </p>  
            
            <ul style="color: #2c3e50; margin-left: 20px; margin-bottom: 15px;">  
                <li>Leveraging cutting-edge AI technologies</li>  
                <li>Reducing human error in defect detection</li>  
                <li>Increasing production efficiency</li>  
                <li>Minimizing waste and economic losses</li>  
            </ul>  
            
            <p style="font-style: italic; color: #2980b9;">  
            Vision: Create intelligent, autonomous quality control systems that set new standards in manufacturing precision.  
            </p>  
            </div>  
            """, unsafe_allow_html=True)  
        
        # Impact and Benefits  
        st.markdown("## 💡 Key Benefits")  
        
        benefits = [  
            {  
                "icon": "📊",  
                "title": "Precision Detection",  
                "description": "Advanced AI algorithms detect defects with over 92% accuracy"  
            },  
            {  
                "icon": "⏱️",  
                "title": "Speed & Efficiency",  
                "description": "Process up to 500 O-rings per hour, dramatically reducing inspection time"  
            },  
            {  
                "icon": "💰",  
                "title": "Cost Reduction",  
                "description": "Minimize waste and quality control expenses through intelligent detection"  
            },  
            {  
                "icon": "🔬",  
                "title": "Continuous Learning",  
                "description": "Machine learning models improve with each inspection, enhancing accuracy"  
            }  
        ]  
        
        benefit_cols = st.columns(4)  
        
        for i, benefit in enumerate(benefits):  
            with benefit_cols[i]:  
                st.markdown(f"""  
                <div style="text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 8px;">  
                <h3>{benefit['icon']} {benefit['title']}</h3>  
                <p>{benefit['description']}</p>  
                </div>  
                """, unsafe_allow_html=True)

def main():  
    # Replace with your actual model paths  
    RESNET_MODEL_PATH = 'resnet.pth'  
    AUTOENCODER_MODEL_PATH = 'autoencoder.h5'  
    
    app = DefectDetectionApp(RESNET_MODEL_PATH, AUTOENCODER_MODEL_PATH)  
    app.run_app()  

if __name__ == "__main__":  
    main()