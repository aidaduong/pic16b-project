from flask import Flask, request, render_template_string

app = Flask(__name__) #creating a web app

class ConditionInfo: #a class to define the conditions 
    def __init__(self, name, description, cause, ingredients, tips, when_to_seek):
        self.name = name
        self.description = description
        self.causes = cause
        self.ingredients = ingredients
        self.tips = tips
        self.help = when_to_seek


#conditions pulled from dataset 1 and 2
CONDITIONS = {}

def add_condition(key, name, description, cause, ingredients, tips, when_to_seek):
    """helper to add each condition without having to type it multiple times"""
    CONDITIONS[key] = ConditionInfo(name=name, description=description, cause=cause, ingredients=ingredients, tips=tips, when_to_seek=when_to_seek,)

#from skin_condition_knowledge_new.csv
add_condition(key="cystic_acne", name="Cystic acne", description="Large, red, painful breakouts deep in the skin.",cause="Hormonal changes, some medicines, and very inflamed clogged pores.",ingredients=[
"Oral antibiotics", "Birth control pills", "Benzoyl peroxide", "Retinoids", "Isotretinoin", "Spironolactone",], tips=["Wash face twice a day with a gentle cleanser", "Do not scrub or pop cysts", "Use non-comedogenic (non-pore-clogging) products", "Limit sun exposure and use sunscreen", "Try to reduce stress and get enough sleep",], when_to_seek="See a dermatologist if cysts are painful, keep coming back, or cause scarring.",)

#blackhead condition and description
add_condition(key="blackheads", name="Blackheads", description="Small dark spots that form when pores are clogged but stay open at the surface.", cause="Oil buildup, dead skin, and clogged hair follicles.", ingredients=["Salicylic acid", "Retinoid creams or lotions",], tips=["Wash face regularly", "Avoid harsh scrubs and very drying products", "Use alcohol-free skincare", "Change pillowcases often", "Keep hair clean and off the face if oily",], when_to_seek="See a dermatologist if blackheads keep coming back or worsen despite good skincare.",)

#inflammatory acne condition and description
add_condition(key="inflammatory_acne", name="Inflammatory acne", description="Red, swollen, sore bumps that may contain pus.",cause="Clogged pores and ruptured pore walls that trigger inflammation.", ingredients=["Azelaic acid", "Benzoyl peroxide", "Niacinamide", "Retinoids (like tretinoin or adapalene)", "Salicylic acid", "Topical antibiotics",], tips=["Avoid touching or picking at pimples", "Use non-comedogenic products", "Avoid harsh scrubs", "Gently cleanse the skin and keep a simple routine", "Limit very sugary foods if they seem to worsen breakouts",], when_to_seek="Get help if inflammation is persistent, worsening, or if nodules/cysts form.",)

#post inflammatory hyperpigmentation condition and description
add_condition(key="pih", name="Post-inflammatory hyperpigmentation (PIH)", description="Dark or discolored spots that appear after inflammation or injury.", cause="Extra melanin made in the skin after it has been irritated or damaged.", ingredients=["Hydroquinone", "Azelaic acid", "Cysteamine", "Vitamin C", "Tretinoin", "Corticosteroids", "Glycolic acid", "Kojic acid",], tips=["Use SPF 50+ sunscreen every day on exposed areas", "Avoid picking or scratching spots", "Use gentle, non-irritating skincare", "Consider camouflage makeup if desired",], when_to_seek="Seek help if pigmentation worsens, does not fade, or causes emotional distress.",)

#closed comedones condition and description
add_condition(key="closed_comedones", name="Closed comedones (whiteheads)", description="Small, skin-colored bumps under the surface of the skin.", cause="Extra oil, dead skin, and pore-blocking skincare or makeup.", ingredients=["Salicylic acid", "Retinoids (tretinoin, adapalene)", "Azelaic acid", "Niacinamide",], tips=["Cleanse gently twice a day", "Avoid heavy or oily skincare and makeup", "Use non-comedogenic products", "Introduce exfoliating products slowly to avoid irritation",], when_to_seek="See a professional if comedones keep increasing, become inflamed, or do not improve.",)

#acne vulgaris condition and description
add_condition(key="acne_vulgaris", name="Acne vulgaris", description="Common acne with blocked pores, inflammation, and breakouts.", cause="Excess oil, clogged pores, bacteria, and inflammation.", ingredients=["Benzoyl peroxide", "Salicylic acid", "Topical retinoids", "Oral antibiotics", "Hormonal therapy (in some cases)"], tips=["Wash face twice daily with a gentle cleanser", "Avoid picking or squeezing pimples", "Use non-comedogenic makeup and skincare", "Keep a consistent routine and be patient"], when_to_seek="See a dermatologist if acne is severe, leaves scars, or does not improve after several weeks.")

#rosacea condition and description
add_condition(key="rosacea", name="Rosacea", description="Chronic redness on the face, sometimes with visible blood vessels or bumps.", cause="Triggered by things like sun, spicy foods, alcohol, temperature changes, or stress.", ingredients=["Topical metronidazole", "Azelaic acid", "Ivermectin cream", "Oral antibiotics for inflammation"], tips=["Use gentle cleansers and moisturizers", "Wear sunscreen every day", "Avoid personal triggers like alcohol or spicy food when possible", "Protect skin from extreme temperatures and wind"], when_to_seek="See a dermatologist if facial redness or bumps persist, worsen, or affect your quality of life.")

#sebaceous cyst condition and description
add_condition(key="sebaceous_cyst", name="Sebaceous cyst", description="A slow-growing, usually harmless lump under the skin.", cause="Blocked hair follicle or oil gland, sometimes after injury.", ingredients=["Warm compresses (for comfort)"], tips=["Avoid squeezing or popping the cyst", "Keep the area clean", "Use warm compresses to help drainage and reduce swelling"], when_to_seek="Seek care if the cyst becomes red, painful, grows quickly, or shows signs of infection.")

#folliculitis condition and description
add_condition(key="folliculitis", name="Folliculitis", description="Small red or pus-filled bumps around hair follicles.", cause="Usually a bacterial infection, but also friction, sweat, or ingrown hairs.", ingredients=["Topical antibiotics", "Antiseptic washes (like chlorhexidine)", "Oral antibiotics for widespread cases"], tips=["Keep the area clean and dry", "Avoid tight clothing over the area", "Avoid shaving the irritated area until it heals"], when_to_seek="See a doctor if the area is widespread, very painful, or keeps coming back.")

#milia condition and description
add_condition(key="milia", name="Milia", description="Tiny hard white bumps under the skin where dead skin gets trapped in pores.", cause="Dead skin cells trapped beneath the surface of the skin.", ingredients=["Gentle exfoliation", "Retinoids", "Chemical exfoliants (AHA/BHA)"], tips=["Avoid heavy creams around eyes", "Do not try to pop milia", "Use gentle exfoliating products"], when_to_seek="See a dermatologist if bumps spread or do not improve.")

#eczema condition and description
add_condition(key="eczema", name="Eczema (Atopic Dermatitis)", description="Red, itchy, dry patches on the skin that come and go.", cause="Skin barrier issues that let irritants or allergens in.", ingredients=["Fragrance-free moisturizers", "Hydrocortisone (mild steroid)", "Colloidal oatmeal creams"], tips=["Moisturize daily", "Avoid hot showers", "Use gentle soaps", "Identify flare triggers"], when_to_seek="Seek care if rashes are painful, widespread, or keep flaring up.")

#keratosis condition and description
add_condition(key="keratosis", name="Keratosis Pilaris", description="Small rough bumps (like chicken skin) on arms, cheeks, or thighs.", cause="A buildup of keratin blocking hair follicles.", ingredients=["Lactic acid", "Urea cream", "Salicylic acid", "Gentle exfoliation"], tips=["Moisturize after showering", "Avoid harsh scrubbing", "Use gentle chemical exfoliants"], when_to_seek="See a dermatologist if bumps become very red, itchy, or painful.")


#writing the actual quiz questions with a dictionary with results

QUESTIONS = [{"id": "lesion_look", "text": "What do your CURRENT breakouts mostly look like? (Select ALL that apply)", "options": {"a": ("Small dark dots / blackheads", {"acne": 3}), "b": ("Small white/skin-colored bumps", {"acne": 2, "milia": 2}), "c": ("Red inflamed pimples", {"acne": 4}), "d": ("Deep painful lumps", {"acne": 5}), "e": ("Flat dark marks afterward", {"acne": 2, "keratosis": 1}), "f": ("General redness + bumps", {"rosacea": 4})}},
    {"id": "pain_level", "text": "How painful or tender are your breakouts? (Select ALL that apply)", "options": {"a": ("Not painful", {"milia": 2, "keratosis": 1}), "b": ("Mild discomfort", {"acne": 2}), "c": ("Often very painful", {"acne": 4})}},
    {"id": "location", "text": "Where do you mostly notice your skin concerns? (Select ALL that apply)", "options": {"a": ("Mostly T-zone", {"acne": 3}), "b": ("Cheeks/sides of face", {"rosacea": 3, "acne": 1}), "c": ("Face + chest/back", {"acne": 4}), "d": ("Scattered patches on body (arms/legs)", {"eczema": 4})}},
    {"id": "after_heal", "text": "After a spot heals, what remains? (Select ALL that apply)", "options": {"a": ("Almost nothing", {"milia": 1}), "b": ("Dark marks that fade slowly", {"acne": 3, "keratosis": 1}), "c": ("Indents or raised scars", {"acne": 4}), "d": ("Dry, flaky patches", {"eczema": 3, "keratosis": 1})}},
    {"id": "redness_triggers", "text": "Do you experience redness triggered by heat/spicy food/alcohol? (Select ALL that apply)", "options": {"a": ("Yes, I flush easily", {"rosacea": 4}), "b": ("Redness mostly around pimples", {"acne": 2}), "c": ("Redness mainly on dry/itchy patches", {"eczema": 3}), "d": ("Rarely any redness", {"milia": 1, "keratosis": 1})}},
    {"id": "routine_goal", "text": "What is your MAIN skincare goal? (Select ALL that apply)", "options": {"a": ("Reduce painful breakouts", {"acne": 4}), "b": ("Clear clogged pores / texture", {"acne": 3, "milia": 2}), "c": ("Fade dark marks / spots", {"acne": 2, "keratosis": 2}), "d": ("Calm redness/sensitivity", {"rosacea": 3, "eczema": 2}), "e": ("Soothe itch / dryness", {"eczema": 4})}}]




#this code creates the quiz page, with each question and checkboxes (so user can select multiple things) for the answers

QUIZ_HTML = """
<h1>Skin Quiz</h1>
<p>Select anything that applies</p>

<form action="/result" method="post"> <!-- submit takes u to results -->
  
  {% for q in questions %} 
    <h3>{{ q.text }}</h3> 

    {% for key, opt in q.options.items() %}
      <label>
        <input type="checkbox" name="{{ q.id }}" value="{{ key }}">
        {{ opt[0] }} 
      </label><br>
    {% endfor %}
    <br>
  {% endfor %}

  <button type="submit">submit</button>
</form>
"""



# This page shows the results after u press submit :D
RESULT_HTML = """
<h1>You might have: {{ primary.name }}</h1>

<h3>What that means</h3>
<p>{{ primary.description }}</p>

<h3>Why it can happen</h3>
<p>{{ primary.causes }}</p>

<h3>What usually helps</h3>
<ul>
  {% for ing in primary.ingredients %}
    <li>{{ ing }}</li>
  {% endfor %}
</ul>

<h3>Simple everyday things to try</h3>
<ul>
  {% for t in primary.tips %}
    <li>{{ t }}</li>
  {% endfor %}
</ul>

<h3>When to get it checked</h3>
<p>{{ primary.help }}</p>

<hr>

<h2>Another possibility: {{ secondary.name }}</h2>
<p>Skin can have more than one thing going on at the same time, so this might overlap with your main result.</p>

<p>Quick description:</p>
<p>{{ secondary.description }}</p>

<br>
<a href="/">Take the quiz again</a>

<p style="font-size:12px; color: gray; margin-top: 20px;">
This quiz is NOT a medical diagnosis.
</p>
"""



#actually giving the diagnosis

def add_points_for_label(scores, label, amount):
    """
    label = thing like 'acne' or 'rosacea.' basically add points to any condition which key has that word
    """
    
    for key in scores.keys():
        if label in key:   #like acne in the term cystic_acne
            scores[key] += amount



#add the points and give first and second diagnoses

@app.route("/")
def index():
    #opening the quiz page
    return render_template_string(QUIZ_HTML, questions=QUESTIONS)

@app.route("/result", methods=["POST"])
def result():
    #start every condition at 0 points
    scores = {key: 0 for key in CONDITIONS.keys()}

    #check what the user picked for each question
    for question in QUESTIONS:
        selected_choices = request.form.getlist(question["id"])

        for choice_key in selected_choices:
            if choice_key not in question["options"]:
                continue  #skip if user enters an option that we dont recognize or something

            option_text, condition_scores = question["options"][choice_key]

            #total the points linked to that label
            for label, points in condition_scores.items():
                add_points_for_label(scores, label, points)

    #figure out which conditions got the highest points
    score_list = list(scores.items())  #taking the score dictionary and turning it into a list
    sorted_scores = sorted(score_list, key=lambda item: item[1], reverse=True)  #having it show the diagnosis with the highest points first as the primary diagnosis


    if not sorted_scores or sorted_scores[0][1] == 0:
        #edge case for if user doesn't pick 
        all_keys = list(CONDITIONS.keys())
        primary_key = all_keys[0]
        secondary_key = all_keys[1] if len(all_keys) > 1 else all_keys[0]
    else:
        primary_key = sorted_scores[0][0]  #top scored for first diagnosis
        secondary_key = sorted_scores[1][0] if len(sorted_scores) > 1 else sorted_scores[0][0]  #finding the second diagnosis

    #showing the actual conditions on the page along with the treatment
    primary_condition = CONDITIONS[primary_key]
    secondary_condition = CONDITIONS[secondary_key]

    return render_template_string(RESULT_HTML, primary=primary_condition, secondary=secondary_condition)


#running final app
if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
