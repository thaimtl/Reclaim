"""Writes the recording prompts for sessions 1-3 and checks them (no duplicates, no overlap with the test set)."""
import re

S1 = """Hi, it's nice to meet you.
Thanks for calling me today.
Can you hear me okay?
I'm doing pretty well, how about you?
My name is Austin and I study computer science.
I'm calling from the hackathon right now.
It's been a long day but a good one.
We barely slept last night.
I had pizza for dinner again.
Do you want to grab coffee later?
I think the weather is getting colder.
It rained all morning here.
I'll be there in about ten minutes.
Let me check my calendar first.
That sounds like a great plan.
I'm not sure I understand the question.
Could you say that one more time?
I really appreciate your help.
We built this in less than two days.
My team is sitting right next to me.
I love working on hard problems.
This is my first time at this event.
I'm a little nervous but excited.
What do you do for work?
How long have you been a judge here?
I grew up in a small town.
My family lives far away from here.
I miss my mom's cooking.
I usually wake up at seven.
I go to the gym three times a week.
My favorite food is ramen.
I don't really like spicy food.
Can we meet after lunch tomorrow?
The bus was late this morning.
I forgot my charger at home.
My phone battery is almost dead.
Please send me the link when you can.
I'll text you the address.
We should take a short break.
I need a glass of water.
It's way too loud in here.
The room is pretty crowded today.
Everyone here seems really friendly.
I met a lot of interesting people.
We talked about music for an hour.
I play the piano a little bit.
I used to play basketball in high school.
My brother is better at sports than me.
I have two sisters and one brother.
Our dog is named Max.
We adopted him from a shelter.
He loves going to the beach.
I want to travel to Japan next year.
Have you ever been to Europe?
I visited Canada last summer.
The flight took almost six hours.
I fell asleep on the plane.
The food at the airport was terrible.
I'd like to order a large coffee.
Can I get that with oat milk?
How much does this cost?
That's a lot cheaper than I expected.
I paid for it with my card.
Keep the change.
I'm saving money for a new laptop.
My old computer is really slow.
The screen is cracked in one corner.
I need to fix my bike this weekend.
The back tire keeps going flat.
I bought a new pair of shoes.
They fit perfectly.
I'm looking for a black jacket.
Do you have this in a bigger size?
I think that color looks good on you.
Let's pick a movie to watch tonight.
I've already seen that one twice.
The ending was a big surprise.
I cried at the end of that movie.
That show is really funny.
I watch it every Friday night.
I'm reading a book about space.
The library closes at nine.
I need to return these books.
I have an exam on Monday.
I studied for six hours yesterday.
Math was always my best subject.
I'm terrible at drawing.
I want to learn how to cook.
Can you teach me that recipe?
Add a pinch of salt and pepper.
Bake it for about twenty minutes.
The bread smells amazing.
Pass me the butter please.
I'll make breakfast tomorrow.
Do you want eggs or pancakes?
I'll have the pancakes with fruit.
Let me know if you need anything.
I'm happy to help out.
That's very kind of you.
No problem at all.
Don't worry about it.
I'm sorry I'm late.
Traffic was really bad.
I took the wrong exit.
The map sent me the long way.
We finally made it.
The parking lot was full.
I found a spot on the street.
The meeting starts at three.
Can we push it to four instead?
I'll bring my notes.
I wrote everything down.
Let's go over the plan again.
The first step is the hardest.
We tested it a hundred times.
It works much better now.
We fixed the bug this morning.
The code is finally clean.
I learned a lot from my mentor.
She gave me really good advice.
He always asks the best questions.
I want to start my own company someday.
Building things makes me happy.
I like solving problems for people.
My grandfather lost his voice after surgery.
He used a buzzer to talk.
It sounded very robotic.
I want to give people their voice back.
Everyone deserves to be heard.
Talking is something we take for granted.
I can't imagine not being able to speak.
It must be very frustrating.
Thank you for listening to me.
What made you want to be a judge?
Where are you calling from?
Is it sunny where you are?
It's almost midnight here.
I should probably go to bed soon.
I'll call you back in a bit.
Talk to you later.
Have a great rest of your day.
Take care of yourself.
See you tomorrow morning.
Good luck with everything.
Congratulations on your new job.
That's amazing news.
I'm so proud of you.
You did a really good job.
I knew you could do it.
Let's celebrate this weekend.
We could go out for dinner.
I know a great place downtown.
They have the best tacos in the city.
The line is usually very long.
We should get there early.
I'll make a reservation for six.
Party of four please.
Can we sit by the window?
The view is beautiful from here.
Look at the sunset.
The sky is bright orange.
I took a picture of it.
I'll send it to you.
My camera is better than my phone.
I bought it last month.
It was on sale.
I got a really good deal.
I think we should leave now.
Grab your bag and let's go.
Did you lock the door?
I left the lights on.
The keys are on the table.
Turn left at the next light.
Keep going straight for a mile.
It's the big building on the right.
We passed it already.
Let's turn around.
I'm getting a little hungry.
Do you have any snacks?
I packed some apples and nuts.
I could eat a whole pizza right now.
Let's order some food.
What are you in the mood for?
Anything is fine with me.
I'll pay this time.
You can get the next one.
This tastes better than I thought.
I'm completely full.
I need a nap.
I didn't sleep well last night.
My neighbors were having a party.
The music was so loud.
I could hear every word.
I wore earplugs but it didn't help.
I'll talk to them tomorrow.
Maybe they'll keep it down.
Life is too short to be angry.
I try to stay positive.
Every day is a new chance.
I believe in hard work.
Practice makes it easier.
Mistakes help you learn.
Don't give up so fast.
Keep trying and you'll get there.
I'm glad we had this talk.
That really means a lot to me.
I feel much better now.
I'm a little tired but happy.
Today was a really good day.
Tomorrow will be even better.
Bring a jacket, it gets cold at night.
The beach is about an hour away.
Pack a towel and some sunscreen.
I'll drive if you want.
My car is parked outside.
Gas prices went up again.
I might take the train instead.
The train is faster during rush hour.
I read the news on the train.
Did you hear about the storm?
They closed the bridge for the day.
Schools are closed tomorrow.
The power went out for an hour.
We lit candles and played cards.
I won every single game.
My sister thinks I cheated.
I promise I didn't cheat.
Let's play again next week.
I'll bring the board game.
The pills are in the bathroom cabinet.
I need to pay my bills by Friday.
Bring a pen and some paper.
The man in the blue coat is my uncle.
I think the bat flew into the barn.
She bought a mat for her yoga class.
Put the pot back on the stove.
My fan stopped working last night.
The van is parked behind the store.
Ten people showed up to the party.
I need a den for my new desk.
Can you get the cat off the couch?
I left my coat in the car.
The game ended in a tie.
He came back with a big smile.
We went down to the river to fish.
The tea is still too hot to drink.
Don't forget to feed the fish before you leave.
I've been looking forward to this conversation all week.
If you have any questions, just let me know and I'll explain.
We spent most of the night trying to make it faster.""".split('\n')

S2 = """Good morning, how did you sleep?
I'm calling to check in on you.
Is now a good time to talk?
Sorry, the connection is a bit bad.
I can hear you much better now.
Give me one second to find a quiet spot.
Okay, I'm back.
What were you saying before?
That makes a lot of sense.
I completely agree with you.
I see your point, but I'm not so sure.
Let me think about that for a moment.
Honestly, I have no idea.
That's a really good question.
I've never thought about it that way.
Can you explain that a little more?
I didn't catch the last part.
So what do you think we should do?
I'd go with the first option.
The second one looks cheaper.
Let's compare them side by side.
I made a list of pros and cons.
We can decide after dinner.
I'm starving, what's for lunch?
How about a sandwich and some soup?
I'll have a turkey sandwich.
Could you put mustard on it?
No onions for me please.
I'm allergic to peanuts.
Is there any milk in this?
I'll just have water, thanks.
The soup is too hot right now.
Let it cool down for a minute.
I spilled coffee on my shirt.
Do you have a napkin?
I need to change before the meeting.
I'm meeting my manager at noon.
She wants to talk about my project.
I think it's going pretty well.
We're a little behind schedule.
We need two more days to finish.
I'll stay late tonight to catch up.
Could you help me with this part?
I'm stuck on the same problem.
I've tried everything I can think of.
Maybe we should ask someone else.
Let's look at it with fresh eyes.
Oh, I see the mistake now.
It was just a missing number.
That was a lot easier than I thought.
Thanks, you saved me hours.
I owe you one.
How was your weekend?
I went hiking with some friends.
We climbed all the way to the top.
My legs are still sore.
The view from the top was incredible.
We saw a deer on the way down.
It started to snow a little.
I wish I had brought gloves.
My hands were freezing.
We stopped for hot chocolate after.
What did you do this weekend?
I mostly stayed home and rested.
I cleaned my whole apartment.
I finally did my laundry.
I found twenty dollars in my pocket.
That made my whole day.
I'm thinking about getting a cat.
Cats are easier than dogs.
My landlord doesn't allow pets.
I might have to move next year.
Rent is getting really expensive.
I'd love to live near the ocean.
I grew up near the mountains.
Winters there were very long.
I learned to ski when I was five.
I broke my arm the first year.
I was in a cast for six weeks.
It itched the whole time.
Do you play any instruments?
I've always wanted to learn guitar.
My roommate plays the drums.
He practices every night at ten.
I bought him headphones for his birthday.
What kind of music do you like?
I listen to a little bit of everything.
I went to a concert last month.
The band played for three hours.
I lost my voice from singing along.
We got home at two in the morning.
I was so tired the next day.
I drank four cups of coffee.
I should really cut back on caffeine.
I'm trying to drink more water.
I started running in the mornings.
I can run three miles now.
My goal is to run a half marathon.
I signed up for a race in May.
Will you come cheer for me?
I'll make a sign for you.
That would be really fun.
Where should we go for vacation?
I've heard Hawaii is beautiful.
Flights are cheaper in the fall.
Let's book it before prices go up.
I'll need to ask for time off.
My boss is pretty flexible.
I haven't taken a day off in months.
I really need a break.
Let's go somewhere warm.
I want to lie on the beach all day.
I'll bring a good book.
Don't forget your passport.
I renewed mine last year.
The photo looks nothing like me.
Everyone says that about their photo.
What time does the store open?
I think it opens at eight.
I need to buy groceries.
We're out of bread and eggs.
Can you pick up some bananas?
Get the ones that are a little green.
I'll make banana bread on Sunday.
My grandma taught me how to bake.
She makes the best apple pie.
I visit her every holiday.
She's turning ninety next month.
We're planning a big party for her.
The whole family is coming.
My cousins are flying in from Texas.
I haven't seen them in five years.
I wonder if they've changed much.
I'll bring my camera to the party.
We should take a family photo.
My uncle always tells the same jokes.
They still make me laugh every time.
Family is really important to me.
I call my parents every Sunday.
My dad just retired last year.
He spends all day in the garden.
He grows tomatoes and peppers.
The tomatoes taste so much better.
My mom is a nurse.
She works long night shifts.
I don't know how she does it.
She's the strongest person I know.
I want to make her proud.
I'm the first in my family to go to college.
It hasn't always been easy.
I had a lot of help along the way.
My teachers believed in me.
One teacher changed my life.
She stayed after class to help me.
I still email her sometimes.
She was so happy when I graduated.
What advice would you give your younger self?
I'd tell myself to worry less.
Most things turn out fine.
It's okay to ask for help.
Nobody has it all figured out.
I'm still learning every day.
I think curiosity is really important.
I ask a lot of questions.
Sometimes too many questions.
My friends tease me about it.
I don't mind, it's who I am.
Are you free on Thursday?
I have a doctor's appointment at ten.
It's just a regular checkup.
I'll be done by eleven.
Let's meet at the coffee shop on Main Street.
I'll get us a table.
Text me when you're on your way.
I'm leaving the house now.
I'm five minutes away.
I'm parking right now.
I see you by the door.
Wow, you cut your hair.
It looks really good.
I needed a change.
I donated the hair I cut.
That's a really nice thing to do.
I try to give back when I can.
I volunteer at the food bank.
We packed three hundred boxes last week.
It feels good to help.
You should come with me next time.
I'd love to.
Just tell me when and where.
It's every Saturday at nine.
Wear comfortable shoes.
We'll be standing most of the day.
Bring a water bottle too.
Did you finish the report?
I sent it last night.
Let me know if anything is missing.
I added the new numbers.
The chart is on page four.
Sales went up by ten percent.
That's better than last year.
The team worked really hard.
We should thank them.
Let's bring them donuts on Monday.
Everybody loves donuts.
I'll get a dozen.
Get some with sprinkles.
Is it supposed to rain tomorrow?
I think it's going to be sunny.
Perfect weather for a picnic.
Let's go to the park.
I'll bring a blanket and some fruit.
Don't forget the frisbee.
We can play until it gets dark.
The park gets busy on weekends.
Let's go early to find a good spot.
Can you believe how fast this year went?
It feels like it just started.
I'm excited for next year.
I have so many plans.
I want to learn a new language.
Maybe Spanish or French.
I took Spanish in high school.
I only remember how to order food.
That's the most important part.
I need to call the bank.
There's a charge I don't recognize.
It was only five dollars.
I still want to check it.
They said it was a mistake.
I'll get the money back in a week.
I lost my wallet once.
Someone found it and mailed it back.
Nothing was missing.
That restored my faith in people.
Mom made beef stew for dinner.
Please put the plates by the sink.
The baby finally fell asleep.
Bob and Pam are coming over later.
My thumb hurts from typing all day.
I think the van needs new brakes.
Five of us went to the movies.
The dentist said my teeth look fine.
I'd like a dozen tulips for my mom.
The kids are playing in the yard.
Grab a bag of ice on your way.
Could you turn the volume down a little?
I keep forgetting to water the plants.
Let me know what you decide and I'll book the tickets.
We could split the cost if that works for you.""".split('\n')

S3 = """Hello, thanks for picking up.
Can you hear me clearly?
I'm doing great today, thanks for asking.
It's really nice to talk to you.
I'm calling from the hackathon.
We've been working since Friday night.
I've had way too much coffee.
I think I slept about three hours.
The food here has been pretty good.
I had a burrito for lunch.
Where are you right now?
Is it cold outside where you are?
It's been windy here all day.
What do you like to do for fun?
I like playing video games with friends.
I also go running when I can.
I'm learning to play the guitar.
My fingers hurt after practice.
Do you have any pets?
I have a cat named Pepper.
She sleeps on my keyboard.
She knocks everything off my desk.
I grew up in California.
My parents still live there.
I visit them during the holidays.
My mom makes amazing dumplings.
I've tried to make them myself.
Mine always fall apart.
What's your favorite movie?
I watched a great movie last week.
It was about a robot that learns to paint.
I didn't expect it to be so sad.
I'd recommend it to anyone.
What did you study in school?
I'm studying computer science.
I really like machine learning.
It's hard but very rewarding.
I want to build things that help people.
My first project was a simple game.
It was full of bugs.
I learned so much from it.
Have you been to a hackathon before?
The energy here is incredible.
Everyone is so creative.
I met a team building a robot arm.
Another team made a music app.
I'm a little nervous right now.
I hope everything works.
We tested it all night.
I'm really proud of my team.
They worked so hard this weekend.
We should all get some sleep tonight.
I can't wait to go home and rest.
What time is it where you are?
It's almost dinner time here.
I'm thinking about getting sushi.
Do you like sushi?
I always order the salmon.
I'd like to visit Japan someday.
I've heard the food is amazing.
I want to see the cherry blossoms.
Maybe I'll go next spring.
Have you traveled much?
I went to Mexico last year.
The beaches were beautiful.
I got a really bad sunburn.
I should have worn more sunscreen.
I'll be more careful next time.
How was your day so far?
That sounds really busy.
You should take a break.
Make sure you drink some water.
I'm glad you could call today.
Thanks for being so patient with me.
Sorry, could you repeat that?
I think I missed what you said.
Oh, that makes sense now.
Yes, I completely agree.
No, I don't think so.
Maybe, I'm not sure yet.
I'll have to think about it.
Let me get back to you on that.
That's a really interesting idea.
I never thought of it like that.
I learned something new today.
It was great talking to you.
I hope we can talk again soon.
Good luck with the rest of the judging.
Enjoy the rest of your evening.
Take care and have a good night.
Bye for now.
I'll send you a message later.
My friend just walked in.
He says hello.
We're going to get dinner together.
I'll call you tomorrow.
Thanks again for everything.
Have a safe trip home.
See you soon.
I'm really happy with how today went.""".split('\n')

SESSIONS = {'session1': S1, 'session2': S2, 'session3_test': S3}
norm = lambda t: re.sub(r'[^a-z0-9 ]', '', t.lower()).strip()
seen, problems = {}, []
for name, lines in SESSIONS.items():
    for i, line in enumerate(lines):
        if not line.strip():
            problems.append(f'{name}:{i + 1} empty')
        key = norm(line)
        if key in seen:
            problems.append(f'duplicate: "{line}" in {seen[key]} and {name}')
        seen[key] = name
assert not problems, '\n'.join(problems)
for name, lines in SESSIONS.items():
    with open(f'{name}.txt', 'w') as f:
        for i, line in enumerate(lines, 1):
            f.write(f'{name.split("_")[0][0]}{name[7] if name.startswith("session") else ""}_{i:03d}\t{line.strip()}\n')
    words = [len(l.split()) for l in lines]
    print(f'{name}: {len(lines)} sentences, {sum(words)} words, {min(words)}-{max(words)} words each')
