# Notes

## **Cracking the Coding Interview**
- If you haven't heard back from a company within 3 - 5 business days after your interview, check in (politely) with your recruiter
- It is difficult to move from an SDET position to a dev position; consider targeting just developer roles or moving within 1 - 2 years of landing an SDET role

### BIG O:
- O (big O): In academia, big O describes an upper bound on time; an algorithm that is O(n) can also be O(n^2) or worse
- Ω (big omega): In academia, Ω describes the lower bound on time; an algorithm that is Ω(n) can also be Ω(1) or better
- Θ (big theta): In academia, Θ describes both big O and big omega; an algorithm is Θ(n) if it is both O(n) and Ω(n)

### Behavioral Question Strategies:
#### **Structuring Your Answer**
- **Nugget First**: Start your response with a "nugget" that succinctly describes what your response will be about
- **S.A.R.**: Situation, Action, and Result; Outline the situation first, then explain the actions you took and follow lastly with the result 

#### Technical Questions Walkthrough
- Listen: Pay Attention to any information in the problem description; Use BUD optimization (Bottlenecks, Unnecessary Work, Duplicated Work) 
- Example: Debug your example - make sure that this is not too small or doesn't have any special cases
- Brute Force: Get a brute force solution as soon as possible; State a naive algorithm and its runtime, then optimize from there
- Optimize: Walk through your brute force with BUD optimization - then get some ideas around why your algorithm is inefficient or why it fails
- Walk Through: Talk through your optimal solution approach in detail
- Implement: Write beautiful code - modularize your code from the beginning and refactor to clean up anything that isn't pretty
- Test: Do a conceptual test first by walking through your code; then highlight unusual or non-standard code; then finish by doing small test cases and maybe larger discuss special cases and edge cases

If you find any bugs, fix them carefully.

<!-- #### **Tell Me About Yourself**: -->
<!-- My name is Leo and I am an ex-investment banker turned software engineer. My background was initially in finance; I completed my undergrad at Baruch College and spent the first few years out of college working as an investment banker. When I first started my career in finance, I had a few key values that I felt were important to me in a job: I had to be able to develop myself technically, and I had to have an impact through my work. While my career in finance did cover those two values, I felt as if I was dramatically slowing in my technical skill development as there are only so many ways to value an asset; and I also felt that my work had a limited scope of impact (mostly executives and investors cared).

Although I was promoted early in my career and had an upward trajectory, I decided to still make the move to software engineering after networking with several people in tech. I decided to leave the finance industry at the end of 2021 and began a full stack software engineering bootcamp at AppAcademy shortly after. I graduated in the summer of 2022 and I have been developing my personal projects and studying.

Outside of developing my personal apps, I participate in weekly coding competitions and assessments to develop my problem solving skills. I also have hobbies such as playing competitive first-person shooters (Valorant) and high fashion. 

I'm here today because I want to develop myself and bring an impact through my work. I have always loved **xyz** and I think I would benefit quite a bit from your **xyz**. -->

#### **Typer Drive**
- **Challenges**: Difficult to learn JavaScript within a week and utilize it to build an animation-based fighting game; utilized 
- **Mistakes / Failures**: Failed at using Canvas to animate health bars on screen; resorted instead to using HTML elements which turned out to be easier to implement and more easy to style with CSS instead of Canvas
- **Enjoyed**: Enjoyed learning about Canvas and figuring out how animations worked - always wondered how a sprite sheet was translated into an on-screen animation and learned exactly how
- **Leadership**: No opportunity for leadership other than just holding myself responsible for making progress on the project
- **Conflicts**: Issue with rendering fluid animations through sprite sheets - realized that I would have to calculate the dimensions of each sprite sheet image and have Canvas render only a portion of each sprite sheet
- **What I Would Change**: Would definitely change the UI / UX of the website and add more features; initially went for a simple look as to not have to 

#### **Thiscord**
- **Challenges**: Learning React components along with WebSocket API was very challenging and difficult to overcome; After much research I was able to learn that Ruby on Rails had a way to setup WebSocket connections easily and thus utilized that to eventually have message functionality
- **Mistakes / Failures**: Realized that my initial run-through of the project was very messy - felt like there were so many React components that were redundant and realized that my code was very inefficient / had a lot of workarounds; eventually returned to the project post-graduation and cleaned it up after learning from my mistakes
- **Enjoyed**: Enjoyed styling the splash page a lot - although the simplest part of the project, I felt like I spent the longest time here because Discord's splash page is very detailed and had a ton of buttons with clean functionality 
- **Leadership**: No opportunity for leadership again but because my project wasn't finished within the allotted time, I communicated well with my career coach on expectations for its completion and eventually was greenlit
- **Conflicts**: Realized many conflicts with WebSocket API - namely that the resources behind Ruby on Rails and WebSocket API documentation felt lacking; I had to experiment quite a bit to get a better understanding of what my components were returning and how my messages would interact and eventually found plenty of resources (technical coaches and others who have completed similar projects) to assist
- **What I Would Change**: Would definitely make it more clear what elements do or do not work on the application - since this project was really just a demo of my full stack capability, there were plenty of features that were not actually implemented but shown on the page (for cloning purposes)

#### **LineAlert**
- **Challenges**: Working with a team of four software engineers to design, develop, and publish a full stack application based on technology we have never used prior (MERN stack); As the team lead, I was challenged with not only leading the group towards the collective goal but also making sure that each team member was responsible and timely in their work
- **Mistakes / Failures**: Realized that our frontend UI / UX design could have been a lot better after discussing with a Google frontend designer; Received a lot of insight of how the page could have looked and why things were better in those ways
- **Enjoyed**: Working within a team and collaborating with others; this was my first opportunity to taste what software engineering was really about and I was pleased to find that it was exciting, collaborative, and thoroughly enjoyable to create an application with others and debug issues together
- **Leadership**: Had many opportunities to exercise my leadership by leading and directing the team on multiple design decisions, including what our backend would include in its database, how it would be structured, and how it would interact with our frontend components and how the website should be designed, what we should include, and how we want the user to interact / utilize the website
- **Conflicts**: Because of the scope of the project and the deliverables we promised, we were very limited on time to get through our MVPs. While no one in the group was difficult to work with, the overall enthusiasm to create the application sometimes got in the way of focusing on core components and features that were promised; I made sure to constantly focus the group on our core features before allowing the team to work on other areas that they felt would be beneficial to the user experience
- **What I Would Change**: If I could go back in time, I would reach out to my network of designers and software engineers first to ask about design (both for systems and frontend) recommendations. Since the timeline was short and my experience was lacking, I felt like we made several design decisions that we would later regret (i.e., unnecessary backend properties that we would never use / frontend design that could have been improved). If I had asked my friends earlier and spent more time contemplating a better overall design, I think we could have had a much better project

## Python:
- Truthy values are values that evaluate to True in a bocept 0 in eolean context
- Falsey values are values that evaluate to False in a boolean context
- Falsey values include empty sequences (lists, tuples, strings, dictionaries, sets), zero in every numeric type, None, and False
- Truthy values include non-empty sequences, numbers (exvery numeric type), and basically every value that is not falsey

Deque (doubly ended queue) 
Implemented using the module "collections"; Deque is preferred over a list in the cases where we need quicker append and pop operations from both ends of the container, as deque provides an O(1) time complexity for append and pop operations as compared to list which provides O(n) time complexity for the same operations
- deque.append(): Used to insert the value in its argument to the right end of the deque
- deque.appendleft(): Used to insert the value in its argument to the left end of the deque
- deque.pop(): Used to delete an argument from the right end of the deque
- deque.popleft(): Used to delete an argument from the left end of the deque

Statistics (Python Library)
You can import statistics library to call statistics-based functions such as mean, median, etc.
- mean(): Average of data
- fmean(): Fast, floating point arithmetic mean
- median(): Middle value of data
- median_low(): Low median of data (first quartile)
- median_high(): High median of data (third quartile)
- mode(): Single mode (most common value) of discrete or nominal data
- covariance(): Sample covariance for two variables
- correlation(): Pearson's correlation coefficient for two variables
- linear_regression(): Slope and intercept for simple linear regression

Built-in Data Types:
- **Lists** are used to store multiple items in a single variable; they are ordered, changeable, indexed, and allow duplicate values (e.g., list = ["apple", "banana", "cherry"])
- **Set** is a collection which is unordered, unchangeable, and unindexed; Note that while the items are unchangeable, they can be removed and added (e.g. set = {"apple", "banana", "cherry"})
- **Tuple** is a collection which is ordered and unchangeable, they are written with round brackets (e.g. tuple = ("apple", "banana", "cherry"))
- **Dictionary** is used to store data values in key:value pairs; it is a collection which is ordered (as of Python v3.7), changeable, and do not allow duplicate keys (e.g., dictionary = {"brand": "Ford", "model": "Mustang", "year": 1964})