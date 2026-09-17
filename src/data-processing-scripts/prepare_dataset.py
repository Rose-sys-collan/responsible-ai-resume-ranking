import pandas as pd 
import re
from datetime import datetime

def convert_to_nationality_code(location):
    if pd.isna(location):
        return "N/A"
    location = str(location).lower().strip()
    
    # Dictionary mapping locations to country codes
    country_mapping = {
        # North America
        'united states': 'US', 'usa': 'US', 'america': 'US',
        'canada': 'CA',
        'mexico': 'MX',
        
        # Europe
        'united kingdom': 'GB', 'uk': 'GB', 'britain': 'GB', 'england': 'GB',
        'france': 'FR',
        'germany': 'DE',
        'italy': 'IT',
        'spain': 'ES',
        'netherlands': 'NL',
        'ireland': 'IE',
        'scotland': 'GB',
        'russia': 'RU',
        'poland': 'PL',
        'sweden': 'SE',
        'norway': 'NO',
        'denmark': 'DK',
        'finland': 'FI',
        
        # Asia
        'india': 'IN', 'mumbai': 'IN', 'delhi': 'IN', 'bangalore': 'IN', 'hyderabad': 'IN',
        'china': 'CN', 'beijing': 'CN', 'shanghai': 'CN',
        'japan': 'JP', 'tokyo': 'JP',
        'south korea': 'KR', 'korea': 'KR',
        'singapore': 'SG',
        'malaysia': 'MY',
        'indonesia': 'ID',
        'philippines': 'PH',
        'vietnam': 'VN',
        'thailand': 'TH',
        'pakistan': 'PK',
        'bangladesh': 'BD',
        'sri lanka': 'LK',
        'nepal': 'NP',
        
        # Middle East
        'united arab emirates': 'AE', 'uae': 'AE', 'dubai': 'AE',
        'saudi arabia': 'SA',
        'qatar': 'QA',
        'kuwait': 'KW',
        'oman': 'OM',
        'bahrain': 'BH',
        'israel': 'IL',
        
        # Australia/Oceania
        'australia': 'AU', 'sydney': 'AU', 'melbourne': 'AU',
        'new zealand': 'NZ',
        
        # Africa
        'south africa': 'ZA',
        'nigeria': 'NG',
        'kenya': 'KE',
        'egypt': 'EG',
        'morocco': 'MA',
        'ghana': 'GH',
        'ethiopia': 'ET',
        
        # South America
        'brazil': 'BR',
        'argentina': 'AR',
        'chile': 'CL',
        'colombia': 'CO',
        'peru': 'PE',
        'venezuela': 'VE'
    }
    
    # Check for major cities and regions
    city_to_country = {
        'new york': 'US', 'california': 'US', 'texas': 'US', 'florida': 'US',
        'london': 'GB', 'manchester': 'GB', 'liverpool': 'GB',
        'paris': 'FR', 'lyon': 'FR',
        'toronto': 'CA', 'vancouver': 'CA', 'montreal': 'CA',
        'sydney': 'AU', 'melbourne': 'AU',
    }
    
    # Try to match the location with country mapping
    for loc, code in country_mapping.items():
        if loc in location:
            return code
            
    # Try to match with cities
    for city, code in city_to_country.items():
        if city in location:
            return code
    
    return "N/A"

df = pd.read_csv("SYRA_Generated_Resumes_dataset.csv", encoding='utf-8-sig')
df = df.drop(columns = "application_name")

#these split for new break for pipe or new line or the word "title"

df[['name', 'job_title']] = (
    df['str_resume']
      .str.split(r'\r?\n|\||(?i:\btitle\b)', n=1, expand=True, regex=True)
      .apply(lambda s: s.str.strip())
)

#splits for new line for pipe character
df[['job_title', 'email']] = df['job_title'].str.split(
    r'\r?\n|\|', n=1, expand=True, regex=True
)

df[['email', 'phone_number']] = df['email'].str.split('|', n=1, expand=True)

df[['phone_number', 'location']] = df['phone_number'].str.split('|', n=1, expand=True)

#these split for targey case insensitive words
df[['location', 'job_description']] = df['location'].str.split(r'(?i)(?=summary|professional summary)', n=1, expand=True) 
df[['job_description', 'skills']] = df['job_description'].str.split(r'(?i)(?=skills)', n=1, expand=True)
df[['skills', 'experience']] = df['skills'].str.split(r'(?i)(?=experience|work experience)', n=1, expand=True)
df[['experience', 'education']] = df['experience'].str.split(r'(?i)(?=education)', n=1, expand=True)
df[['education', 'other']] = df['education'].str.split(r'(?i)(?=projects|languages)', n=1, expand=True)


df = df.drop(columns = "str_resume")

df['nationality'] = df['location'].apply(convert_to_nationality_code)

df = df.drop(columns = "email")
df = df.drop(columns = "phone_number")
df = df.drop(columns = "other")


def _experience_to_decade_label(experience_text):
    """Simple heuristic: extract years of experience and map to a decade label like '40-49'.

    - Look for patterns like '5 years', '3+ years', '5 yrs' and use that number.
    - Estimate age = 22 + years_of_experience, then map to decade label '20-29', '30-39', etc.
    - If no info found, return 'Unknown'.
    """
    if not isinstance(experience_text, str):
        return 'Unknown'

    m = re.search(r"(\d{1,2})\+?\s*(?:years|yrs|year)", experience_text, flags=re.I)
    if m:
        years_exp = int(m.group(1))
        age = 22 + years_exp
        if age < 0:
            return 'Unknown'
        if age > 99:
            age = 99
        decade = (age // 10) * 10
        return f"{decade}-{decade+9}"

    # fallback: try to find any standalone small integer that plausibly indicates experience
    m2 = re.search(r"\b(\d{1,2})\b\s*(?:years|yrs|year)?", experience_text, flags=re.I)
    if m2:
        years_exp = int(m2.group(1))
        age = 22 + years_exp
        if age < 0:
            return 'Unknown'
        if age > 99:
            age = 99
        decade = (age // 10) * 10
        return f"{decade}-{decade+9}"

    return 'Unknown'

# Create single string column 'age_group' with labels like '30-39'. Unknown => 'Unknown'.
df['age_group'] = df['experience'].apply(_experience_to_decade_label)

# Drop rows where age couldn't be determined
df = df[df['age_group'] != 'Unknown']

# Remap older decades into '70-79' bucket per request
df['age_group'] = df['age_group'].replace({'80-89': '70-79', '90-99': '70-79'})

df['name'] = df['name'].str.lower()
df['job_title'] = df['job_title'].str.lower()
df['experience'] = df['experience'].str.lower()
df['education'] = df['education'].str.lower()
df['skills'] = df['skills'].str.lower()
df['job_description'] = df['job_description'].str.lower()


print(df.head(1000))

df.to_csv("clean_dataset.csv")