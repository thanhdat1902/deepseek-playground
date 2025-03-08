# test_fewshot.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load dataset for examples
dataset = load_dataset("json", data_files="disaster_data.jsonl")
examples = dataset["train"]

# Few-shot prompt with 3 examples
few_shot_prompt = """Below are examples of detailed seismic data analyses. Provide a complete analysis for the given query in the same structured format, including all sections as shown.

Example 1:
Query: event_title: M 5.6 - 10 km NW of San Jose, Costa Rica\nevent_description: A moderate earthquake struck near San Jose, Costa Rica, causing minor shaking and localized damage.\ndisaster_type: Earthquake\nevent_date_time: 2025-03-15 14:22:45 UTC\nevent_location: 10 km NW of San Jose, Costa Rica\nevent_coordinates: [-84.1234, 9.8765, 15.2]\ndisaster_details: magnitude: 5.6, alert_level: Yellow, tsunami_risk: 0, significance_level: 350, felt_reports: 1200, damage_reports: Minor cracks in older buildings, no injuries reported.\nclimate_data: temperature: 22.5°C, windspeed: 12.3 km/h, winddirection: 145°, humidity: 78.4%, precipitation_probability: 15%, cloud_cover: 45.6%, pressure_sea_level: 1012.3 hPa
Analysis: <disaster_analysis>\n1. **Key Data Points Extraction:**\n   - **Event Title**: M 5.6 - 10 km NW of San Jose, Costa Rica\n   - **Disaster Type**: Earthquake\n   - **Date and Time**: 2025-03-15 14:22:45 UTC\n   - **Location**: 10 km NW of San Jose, Costa Rica\n   - **Coordinates**: [-84.1234, 9.8765, Depth: 15.2 km]\n   - **Magnitude**: 5.6\n   - **Alert Level**: Yellow\n   - **Tsunami Risk**: 0\n   - **Significance Level (sig)**: 350\n   - **Weather Conditions**: Temperature: 22.5°C, Windspeed: 12.3 km/h, Humidity: 78.4%, Cloud Cover: 45.6%.\n\n2. **Summary of Key Details:**\n   - A moderate earthquake of magnitude 5.6 occurred near San Jose, Costa Rica.\n   - Minor shaking and localized damage have been reported, with no tsunami risk.\n   - Weather conditions are mild, with moderate humidity and light winds.\n\n3. **Current Situation and Potential Progression:**\n   - The earthquake caused minor cracks in older buildings but no injuries.\n   - Aftershocks are possible but unlikely to be significant due to the moderate magnitude.\n   - Emergency response efforts are underway, with no major disruptions reported.\n\n4. **Historical Patterns and Similar Past Events:**\n   - Costa Rica is located in a seismically active region due to tectonic plate interactions.\n   - Similar magnitude earthquakes in the past have caused localized damage but no major casualties.\n\n5. **Predicted Trajectory and Potential Impacts:**\n   - The earthquake's impact is expected to remain localized, with no widespread damage.\n   - Aftershocks may occur but are unlikely to exceed magnitude 4.0.\n\n6. **Effects on Urban Areas, Wildlife, and Infrastructure:**\n   - Urban areas may experience minor structural damage, particularly in older buildings.\n   - Wildlife is unlikely to be significantly affected.\n   - Infrastructure impacts are minimal, with no major disruptions to utilities or transportation.\n\n7. **Severity Estimation:**\n   - The event is categorized as moderate with a significance level of 350.\n\n8. **Safety Recommendations and Mitigation Strategies:**\n   - Inspect buildings for structural damage, especially older constructions.\n   - Prepare for potential aftershocks and ensure emergency supplies are available.\n   - Public advisories should be issued to keep residents informed.\n\n9. **Weather Influence:**\n   - Mild weather conditions are unlikely to exacerbate the earthquake's impact.\n\n10. **Reliability and Completeness of Data:**\n    - The earthquake data is reliable and complete, with no significant gaps.\n\n11. **Historical Comparisons:**\n    - Similar events in the region have resulted in localized damage but no major crises.\n\n12. **Localized Recommendations and Resources:**\n    - Local authorities should focus on damage assessment and public safety.\n    - Residents should avoid damaged structures and follow official guidance.\n\n13. **Uncertainty Quantification:**\n    - Confidence in predictions is high regarding the localized impact.\n    - Aftershocks introduce minor uncertainty but are unlikely to be severe.\n</disaster_analysis>\n\n<event_summary>\nThis event is a moderate earthquake of magnitude 5.6 located 10 km NW of San Jose, Costa Rica. Occurring on March 15, 2025, it has caused minor damage but poses no tsunami risk or immediate large-scale threats.\n</event_summary>\n\n<detailed_analysis>\nThe earthquake near San Jose, Costa Rica, is moderate with localized impacts. The region's history of seismic activity suggests that the event will not escalate significantly. Mild weather conditions are unlikely to complicate response efforts, and infrastructure impacts are expected to be minimal.\n</detailed_analysis>\n\n<predictions>\nThe earthquake is expected to have localized impacts, with minor structural damage and no significant aftershocks. Emergency response efforts will likely proceed without major disruptions.\n</predictions>\n\n<impacts>\nUrban areas may experience minor structural damage, particularly in older buildings. Wildlife and critical infrastructure are unlikely to be significantly affected.\n</impacts>\n\n<mitigation_strategies>\n- Inspect buildings for structural damage.\n- Prepare for potential aftershocks.\n- Issue public advisories to keep residents informed.\n</mitigation_strategies>

Example 2:
Query: event_title: M 6.2 - 15 km S of Los Angeles, CA\nevent_description: A strong earthquake caused significant shaking and damage in southern LA.\ndisaster_type: Earthquake\nevent_date_time: 2025-03-16 09:30:00 UTC\nevent_location: 15 km S of Los Angeles, CA\nevent_coordinates: [-118.2437, 33.9522, 12.0]\ndisaster_details: magnitude: 6.2, alert_level: Orange, tsunami_risk: 0, significance_level: 620, felt_reports: 5000, damage_reports: Collapsed structures, minor injuries reported.\nclimate_data: temperature: 18.0°C, windspeed: 8.5 km/h, winddirection: 200°, humidity: 65.2%, precipitation_probability: 5%, cloud_cover: 20.1%, pressure_sea_level: 1015.7 hPa
Analysis: <disaster_analysis>\n1. **Key Data Points Extraction:**\n   - **Event Title**: M 6.2 - 15 km S of Los Angeles, CA\n   - **Disaster Type**: Earthquake\n   - **Date and Time**: 2025-03-16 09:30:00 UTC\n   - **Location**: 15 km S of Los Angeles, CA\n   - **Coordinates**: [-118.2437, 33.9522, Depth: 12.0 km]\n   - **Magnitude**: 6.2\n   - **Alert Level**: Orange\n   - **Tsunami Risk**: 0\n   - **Significance Level (sig)**: 620\n   - **Weather Conditions**: Temperature: 18.0°C, Windspeed: 8.5 km/h, Humidity: 65.2%, Cloud Cover: 20.1%.\n\n2. **Summary of Key Details:**\n   - A strong M 6.2 earthquake struck south of Los Angeles.\n   - Significant shaking led to collapsed structures and minor injuries.\n   - Weather is clear, aiding response efforts.\n\n3. **Current Situation and Potential Progression:**\n   - Damage includes building collapses; emergency services are active.\n   - Aftershocks are likely within 48 hours.\n   - Clear weather supports rapid response.\n\n4. **Historical Patterns and Similar Past Events:**\n   - LA’s seismic history includes the 1994 Northridge quake (M 6.7).\n\n5. **Predicted Trajectory and Potential Impacts:**\n   - Widespread damage expected; injuries may increase.\n   - Aftershocks could reach M 5.0.\n\n6. **Effects on Urban Areas, Wildlife, and Infrastructure:**\n   - Urban areas: Structural damage, disrupted services.\n   - Wildlife: Habitat disruption.\n   - Infrastructure: Power outages, road closures.\n\n7. **Severity Estimation:**\n   - High severity (sig: 620).\n\n8. **Safety Recommendations and Mitigation Strategies:**\n   - Evacuate damaged buildings.\n   - Prepare for aftershocks with emergency kits.\n   - Coordinate rapid response teams.\n\n9. **Weather Influence:**\n   - Clear conditions facilitate rescue efforts.\n\n10. **Reliability and Completeness of Data:**\n    - Data is real-time and reliable.\n\n11. **Historical Comparisons:**\n    - Northridge quake suggests potential for significant disruption.\n\n12. **Localized Recommendations and Resources:**\n    - Prioritize rescue in densely populated areas.\n    - Residents avoid freeways and secure homes.\n\n13. **Uncertainty Quantification:**\n    - 85% confidence in damage scope; aftershock timing uncertain.\n</disaster_analysis>\n\n<event_summary>\nM 6.2 earthquake 15 km south of Los Angeles on March 16, 2025. Orange alert, significant damage reported.\n</event_summary>\n\n<detailed_analysis>\nStrong quake causing widespread damage. Historical parallels like Northridge suggest ongoing risks; clear weather aids response.\n</detailed_analysis>\n\n<predictions>\nSignificant damage and aftershocks expected within 48 hours.\n</predictions>\n\n<impacts>\nUrban structural damage, wildlife disruption, infrastructure strain.\n</impacts>\n\n<mitigation_strategies>\n- Reinforce damaged structures.\n- Stock emergency supplies.\n- Coordinate multi-agency response.\n</mitigation_strategies>

Example 3:
Query: event_title: M 3.8 - 8 km E of Anchorage, AK\nevent_description: A light earthquake was felt in Anchorage with no reported damage.\ndisaster_type: Earthquake\nevent_date_time: 2025-03-17 03:15:00 UTC\nevent_location: 8 km E of Anchorage, AK\nevent_coordinates: [-149.8003, 61.2181, 20.5]\ndisaster_details: magnitude: 3.8, alert_level: Green, tsunami_risk: 0, significance_level: 200, felt_reports: 300, damage_reports: None reported.\nclimate_data: temperature: -5.0°C, windspeed: 15.0 km/h, winddirection: 320°, humidity: 90.0%, precipitation_probability: 60%, cloud_cover: 80.0%, pressure_sea_level: 998.5 hPa
Analysis: <disaster_analysis>\n1. **Key Data Points Extraction:**\n   - **Event Title**: M 3.8 - 8 km E of Anchorage, AK\n   - **Disaster Type**: Earthquake\n   - **Date and Time**: 2025-03-17 03:15:00 UTC\n   - **Location**: 8 km E of Anchorage, AK\n   - **Coordinates**: [-149.8003, 61.2181, Depth: 20.5 km]\n   - **Magnitude**: 3.8\n   - **Alert Level**: Green\n   - **Tsunami Risk**: 0\n   - **Significance Level (sig)**: 200\n   - **Weather Conditions**: Temperature: -5.0°C, Windspeed: 15.0 km/h, Humidity: 90.0%, Cloud Cover: 80.0%.\n\n2. **Summary of Key Details:**\n   - A light M 3.8 earthquake occurred east of Anchorage.\n   - No damage reported; green alert issued.\n   - Cold, snowy weather may affect visibility.\n\n3. **Current Situation and Potential Progression:**\n   - Minor shaking felt; no structural damage.\n   - Low risk of aftershocks.\n   - Weather could complicate monitoring efforts.\n\n4. **Historical Patterns and Similar Past Events:**\n   - Anchorage experiences frequent minor quakes.\n\n5. **Predicted Trajectory and Potential Impacts:**\n   - Minimal impact expected; no significant progression.\n\n6. **Effects on Urban Areas, Wildlife, and Infrastructure:**\n   - Urban areas: No notable effects.\n   - Wildlife: Minor disturbance.\n   - Infrastructure: Unaffected.\n\n7. **Severity Estimation:**\n   - Low severity (sig: 200).\n\n8. **Safety Recommendations and Mitigation Strategies:**\n   - Routine monitoring sufficient.\n   - Prepare for cold weather conditions.\n\n9. **Weather Influence:**\n   - Snow and wind may reduce visibility for response teams.\n\n10. **Reliability and Completeness of Data:**\n    - Data is complete and trustworthy.\n\n11. **Historical Comparisons:**\n    - Similar events have had negligible impact.\n\n12. **Localized Recommendations and Resources:**\n    - Maintain weather preparedness.\n    - Monitor USGS updates.\n\n13. **Uncertainty Quantification:**\n    - High confidence in minimal impact prediction.\n</disaster_analysis>\n\n<event_summary>\nM 3.8 earthquake 8 km east of Anchorage, AK, on March 17, 2025. Green alert, no damage.\n</event_summary>\n\n<detailed_analysis>\nLight quake with no significant impact. Cold weather may pose minor challenges; historical trends suggest low risk.\n</detailed_analysis>\n\n<predictions>\nNo significant impacts or aftershocks expected.\n</predictions>\n\n<impacts>\nMinimal effects on urban areas, wildlife, and infrastructure.\n</impacts>\n\n<mitigation_strategies>\n- Continue routine monitoring.\n- Address weather-related visibility issues.\n</mitigation_strategies>

Now, analyze the following query:
Query: event_title: M 6.0 - 7 km W of Los Angeles, CA\nevent_description: Strong quake near city.\ndisaster_type: Earthquake\nevent_date_time: 2025-03-04 10:00:00 UTC\nevent_location: 7 km W of Los Angeles, CA\nevent_coordinates: [-118.2437, 34.0522, 12.0]\ndisaster_details: magnitude: 6.0, alert_level: Yellow, tsunami_risk: 0, significance_level: 600, felt_reports: 3000, damage_reports: Minor structural damage reported.\nclimate_data: temperature: 20.0°C, windspeed: 10.0 km/h, winddirection: 180°, humidity: 60.0%, precipitation_probability: 10%, cloud_cover: 30.0%, pressure_sea_level: 1014.0 hPa
Analysis:"""

inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=1000,
    temperature=0.5,
    top_p=0.95,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response.split('Analysis:')[-1].strip() if 'Analysis:' in response else response}")