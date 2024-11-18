

PROMPTS = {}

PROMPTS["default"] = ""

PROMPTS["direct_answer"] = "Let's answer the question directly."

PROMPTS["step_by_step"] = "Let's think step by step."

PROMPTS["plan_and_solve"] = "Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve the problem step-by-step."

PROMPTS["fact_and_reflection"] = "Let's first identify the relevant information from the long context and list it. Then, carry out step-by-step reasoning based on that information, and finally, provide the answer."

PROMPTS["eval_longbench_qa"] = """
{context}

{input}
"""

PROMPTS["eval_loong"] = """
[Question]
{}

[Gold Answer]
{}

[The Start of Assistant's Predicted Answer]
{}
[The End of Assistant's Predicted Answer]

[System]
We would like to request your feedback on the performance of the AI assistant in response to the user question displayed above according to the gold answer. Please use the following listed aspects and their descriptions as evaluation criteria:
    - Accuracy and Hallucinations: The assistant's answer is semantically consistent with the gold answer; The numerical value and order need to be accurate, and there should be no hallucinations.
    - Completeness: Referring to the reference answers, the assistant's answer should contain all the key points needed to answer the user's question; further elaboration on these key points can be omitted.
Please rate whether this answer is suitable for the question. Please note that the gold answer can be considered as a correct answer to the question.

The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.
Please note that if the assistant's answer and the gold answer fully meet the above criteria, its overall rating should be the full marks (100).
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias.
Then, output a line indicating the score of the Assistant.

PLEASE OUTPUT WITH THE FOLLOWING FORMAT, WHERE THE SCORE IS A SCALE OF 1 TO 100 BY STRICTLY FOLLOWING THIS FORMAT: "[[score]]", FOR EXAMPLE "Rating: [[100]]":
<start output>
Evaluation evidence: your evaluation explanation here, no more than 100 words
Rating: [[score]]
<end output>

Now, start your evaluation:
"""

PROMPTS["summarize"] = """
{context}

Please summarize the above material, briefly outlining its main content in around {num_summary_words} words in a paragraph.
"""

PROMPTS["summarize_facts"] = """
{context}

Please summarize the most important and relevant {num_facts} facts from the material above in order.
Each fact should consist of around {num_fact_words} words, be self-contained, informative, and convey a complete idea. It should offer clear and concise information, ensuring that the concept stands alone and is easy to understand without additional context.
Please only respond with the facts.
"""

PROMPTS["synthesize_reasoning_instruction_direct"] = """
{context}

Please create a question involving reasoning based on the above material.
Please only respond with the question.
"""

PROMPTS["synthesize_reasoning_instruction_direct_few_shot"] = """
{context}

Please create a question involving reasoning based on the above material.

Here are some demonstrations:
{demonstrations}

Please only respond with the question.
Question:
"""

PROMPTS["synthesize_reasoning_instruction_facts_few_shot"] = """
Please create a question involving reasoning based on some key facts of the material.

Here are some demonstrations:
{demonstrations}

Please only respond with the question.
Facts:
{facts}
Question:
"""

PROMPTS["reasoning_instruction_demonstrations_summary"] = [
    {
        "summary": "The text appears to be a collection of biographies of Japanese athletes who have competed in various sports at the Olympic Games and other international competitions. The athletes come from a range of sports, including track and field, swimming, gymnastics, judo, and speed skating, among others. Each biography provides a brief overview of the athlete's career, including their achievements, awards, and notable performances. The biographies also include personal details, such as birthdates, places of birth, and education. The text provides a comprehensive overview of Japanese athletes who have made significant contributions to their respective sports.",
        "instruction": "How many times is Japan mentioned as having won Olympic gold medals in this article?",
        "type": "numerical"
    },
    {
        "summary": "The text provides a collection of biographies and sports results from various Olympic Games, primarily from the 1972 Summer Olympics in Munich, West Germany. It features athletes from different countries and sports, including track and field, swimming, cycling, wrestling, and more. The biographies include birth dates, competition results, and notable achievements for each athlete. The text also includes medal counts and overall results for participating countries, such as Poland and East Germany. Additionally, some entries provide information on athletes who competed in earlier or later Olympics, offering a broader scope of Olympic history.",
        "instruction": "List the top three countries that won the most medals in the 1972 Olympics and provide their medal counts.",
        "type": "numerical"
    },
    {
        "summary": "The provided text includes information about various weightlifters, strongmen, and powerlifters from different countries. It contains brief biographies, achievements, and records of each individual, highlighting their participation in international competitions such as the Olympics, World Championships, and Commonwealth Games. Additionally, it provides details about the careers and personal lives of well-known weightlifters, including Naim Süleymanoğlu, also known as \"Pocket Hercules,\" who is widely regarded as one of the greatest Olympic weightlifters of all time.",
        "instruction": "How many times has Pocket Hercules won the Olympics?",
        "type": "numerical"
    },
    {
        "summary": "The provided text encompasses various topics related to mining and geology. It includes articles about specific mines, such as the Gwalia Gold Mine in Australia, the Willow Creek mining district in Alaska, and the Canonbie Coalfield in Scotland. There are also articles about mining companies, like Hancock Prospecting Pty Ltd in Australia and China National Gold Group Corporation. Furthermore, the text covers general topics, including Australian mining law, uranium mining in Western Australia, and copper mining in the United States. The articles provide historical and geographical context for each mine or company, as well as information about their production and operations. Some articles also discuss environmental concerns and controversies related to mining activities. Additionally, the text includes information about the geology and mineral deposits of specific regions, such as the Lake District in England and the Absaroka Mountains in Wyoming. Overall, the provided text offers a comprehensive overview of various aspects of the mining industry, including its history, geology, and environmental impact. It also highlights the economic importance of mining and the challenges faced by mining companies in different parts of the world.",
        "instruction": "Identify the most profitable gold mine mentioned in the text.",
        "type": "numerical"
    },
    {
        "summary": "The text is a collection of articles about architects, architectural firms, and their works. It includes profiles of various architects, such as Sholl & Fay, Frank L. Packard, and Earl R. Flansburgh, highlighting their notable works and styles. The text also mentions architectural firms like Esenwein & Johnson, Walker & Gillette, and Hentz, Reid & Adler, showcasing their notable projects and designs. In addition, the text covers various topics such as the history of octagon houses, the work of Grosvenor Atterbury, and the design principles of Francis Riley Heakes. It also touches on the biographies of architects like Jean Labatut, Joseph William Royer, and Amos Porter Cutting, highlighting their contributions to the field of architecture. The text also includes information on the Korte Company, a design-build and construction management firm, and Crimson Historians & Urbanists, a group of architectural historians and urbanists. The articles provide a comprehensive overview of the architects' and firms' works, styles, and contributions to the field of architecture.",
        "instruction": "How many architectural styles are mentioned in the text?",
        "type": "numerical"
    },
    {
        "summary": "The article discusses various topics related to explosives, munitions, and military history. It begins with a description of sealed rounds, which are munitions that can be stored for long periods without maintenance. The article then covers different types of artillery, including surface-to-air missiles, anti-tank missiles, and siege artillery. Additionally, it mentions the development of anti-tank rifles and their use in World War I and World War II. The article also touches on the history of explosives, including the invention of gunpowder and its transmission from China to Europe.",
        "instruction": "Was the anti-tank rifle more widely used in World War I or World War II?",
        "type": "numerical"
    },
    {
        "summary": "The material consists of various articles about places and topics related to Arizona, USA. It includes descriptions of populated places such as Little Acres, Manila, and Puertocito, as well as geological features like Boundary Cone. The articles also cover the history and architecture of the Mesa Arizona Temple, the Phoenix Arizona Temple, and the Tonto National Monument. Additionally, there are entries on the economy of Arizona, Jewish people in Arizona, and the San Xavier Indian Reservation. Other topics include the Verde Valley, Catalina Foothills, Tanque Verde, and the town of Florence, with its historic buildings and rich history.",
        "instruction": "How has Arizona’s economic development trended over the years?",
        "type": "temporal"
    },
    {
        "summary": "KSHB-TV, a television station in Kansas City, Missouri, was established in 1970 as KBMA-TV and later renamed to KSHB-TV in 1981. The station was owned by the E. W. Scripps Company and affiliated with NBC, broadcasting news, sports, and entertainment programs. Its history includes a transition from an independent station to a Fox affiliate and later an NBC affiliate. KSHB-TV's news operation has undergone changes, including the introduction of a 9 p.m. newscast and later a full suite of local newscasts. The station has also produced various local programs, such as sports and non-news shows.",
        "instruction": "In chronological order, which companies has KSHB partnered with over the years?",
        "type": "temporal"
    },
    {
        "summary": "The provided text discusses the COVID-19 lockdown in India and the Pfizer-BioNTech COVID-19 vaccine. In India, a nationwide lockdown was implemented on March 24, 2020, to slow the spread of the virus, resulting in a significant economic impact. The lockdown had a mixed effect, with some people facing hardship while others felt safer. The Pfizer-BioNTech vaccine is an mRNA-based vaccine that has been approved for emergency use in several countries. It has been shown to be effective in preventing severe illness and hospitalization due to COVID-19, with a high efficacy rate and a good safety profile.",
        "instruction": "How long after the COVID-19 lockdown in India did the Pfizer-BioNTech COVID-19 vaccine start being distributed?",
        "type": "temporal"
    },
    {
        "summary": "The provided text consists of numerous book summaries, reviews, and analyses across various genres. The books range from horror, mystery, and thriller to science fiction, fantasy, and dystopian novels. Some of the notable works include Dean Koontz's \"Intensity,\" James Dashner's \"The Maze Runner,\" and Jasper Fforde's \"The Eyre Affair.\" The summaries cover a wide range of plotlines, from a college student's encounter with a serial killer in \"Intensity\" to a group of teenagers trapped in a mysterious maze in \"The Maze Runner.\" The Eyre Affair, on the other hand, follows a literary detective as she pursues a master criminal through the world of Charlotte Brontë's \"Jane Eyre.\" The reviews and analyses provide insight into the authors' writing styles, character development, and the effectiveness of their storytelling. Some books, like \"The Maze Runner,\" are praised for their fast-paced action and engaging plots, while others, like \"The Eyre Affair,\" are commended for their originality and unique blend of genres. Overall, the text offers a diverse collection of book summaries, reviews, and analyses that cater to various reading interests and preferences.",
        "instruction": "Which book was published first, Intensity or The Maze Runner?",
        "type": "temporal"
    },
    {
        "summary": "The provided text contains a collection of biographies, historical accounts, and encyclopedic entries on various topics, primarily related to the Ottoman Empire, Turkish history, and Islamic studies. The entries cover a range of subjects, including notable Ottoman statesmen, military leaders, and intellectuals, as well as historical events, cultural institutions, and social classes. The biographical entries include accounts of prominent figures such as Mehmed Necib Pasha, Mustafa Suphi, and Mehmed Esad Saffet Pasha, who played significant roles in shaping Ottoman politics, society, and culture. Other entries focus on historical events, such as the 1835-1858 revolt in Ottoman Tripolitania, the Russo-Ottoman War of 1710-11, and the assassination of Mehmed-beg Kulenović. The text also includes descriptions of cultural institutions, such as the Mahmut Pasha Hamam, the Öküz Mehmed Pasha Caravanserai, and the Dadian family's historical significance. Additionally, there are entries on social classes, including the askeri class, and the grand viziers of the Ottoman Empire. Overall, the text provides a wealth of information on various aspects of Ottoman history, politics, culture, and society,",
        "instruction": "How many years were there between the Tripolitanian revolt and the Russo-Ottoman War?",
        "type": "temporal"
    },
    {
        "summary": "The mutual information (MI) of two random variables is a measure of their mutual dependence. It quantifies the amount of information obtained about one variable by observing the other. MI is a more general concept than correlation coefficient, as it can capture non-linear relationships between variables. The definition of MI involves the joint distribution, marginal distributions, and the Kullback-Leibler divergence. Properties of MI include non-negativity, symmetry, and supermodularity under independence. MI has various applications in information theory, signal processing, and machine learning. Normalized variants and generalizations to more than two variables have also been proposed.",
        "instruction": "Why mutual information is widely used in machine learning?",
        "type": "logical"
    },
    {
        "summary": "The material discussed the lives and achievements of various Chinese historical figures, including Gongsun Zan, Xu Wan, Zhang Zhidong, Qian Hongzong, Xue Ne, Li Sujie, Xue Ji, Wang Kuang, Liu Zixun, Liu Bei, and Emperor Xiangzong of Western Xia. These individuals lived during different Chinese dynasties, such as the Han, Tang, and Yuan, and played important roles in politics, military, and governance. Some, like Gongsun Zan and Zhang Zhidong, were known for their military prowess and administrative skills, while others, like Qian Hongzong and Xue Ne, were notable for their intellectual and artistic achievements.",
        "instruction": "Did Liu Bei and Zhang Zhidong ever meet?",
        "type": "logical"
    },
    {
        "summary": "The text covers various topics related to Canadian politics, including election results, leadership elections, and party standings. The 1948 Saskatchewan general election saw the Co-operative Commonwealth Federation (CCF) government re-elected with a reduced majority. The Liberal Party of Saskatchewan increased its representation in the legislature from 5 to 19 seats. In other provinces, the 2003 Nova Scotia general election saw the Progressive Conservative Party reduced to a minority government, while the New Democratic Party (NDP) increased its seat count. The 2019 Northwest Territories general election resulted in a historic breakthrough for women in politics, with nine women elected to the 19-seat legislature. The text also covers leadership elections, such as the 1961 NDP founding convention, where Tommy Douglas was elected leader, and the 2009 Ontario NDP leadership election, which saw Andrea Horwath become the party's leader. Additionally, the text discusses the Citizens' Assembly on Electoral Reform in Ontario, which recommended the adoption of a mixed-member proportional representation system in 2007. However, the proposal was rejected by 63% of voters in a referendum.",
        "instruction": "Why did the number of women elected in 2019 increase?",
        "type": "logical"
    },
    {
        "summary": "The provided text contains information about various music releases, including songs, albums, and EPs, across different genres and languages. It includes details about artists such as Cascada, Aya Nakamura, Dafina Zeqiri, and Aitana, as well as their respective releases, including songs like \"Glorious\", \"Oublier\", \"Million $\", and \"Teléfono\". The text also covers albums like Papillon by Lara Fabian and Tráiler by Aitana, and EPs like Rema by Rema. Additionally, it mentions music videos, chart performances, and certifications for some of the releases. The text spans multiple years and languages, showcasing a diverse range of music content.",
        "instruction": "What other well-known songs has the artist who performed ‘Glorious’ released?",
        "type": "multi-hop"
    },
    {
        "summary": "The material covers the participation of various countries in the Olympic Games, including Tanzania, Portugal, and Sweden, as well as specific events such as the men's 100m freestyle and the women's synchronized swimming. It also includes information on the 2008 Summer Paralympics, including the participation of countries such as Lithuania and the events of track cycling and road cycling. Additionally, the material covers the 1988 Summer Olympics, including the participation of countries such as the Soviet Union and the events of athletics, cycling, and synchronized swimming. The material provides detailed information on the athletes, events, and results of the Olympic Games.",
        "instruction": "How many gold medals did the country that won the men’s 100m freestyle in 1988 win in total?",
        "type": "multi-hop"
    },
    {
        "summary": "The material covers various topics related to the Catholic Church, including papal documents, ecumenical councils, and the Church's teachings on marriage and slavery. The papal documents discussed include \"Satis Cognitum,\" which emphasizes the unity of the Church, and \"Arcanum,\" which outlines the rule of marriage in the late 19th century. The ecumenical councils mentioned include the Synod of Hippo, which approved a Christian Biblical canon, and the Council of London, which dealt with issues related to clerical celibacy and the slave trade. The Church's teachings on marriage and slavery are also discussed, with a focus on the importance of unity and the condemnation of slavery.",
        "instruction": "Who was the Pope of the Church that issued the document outlining the rules of marriage?",
        "type": "multi-hop"
    },
    {
        "summary": "The provided text covers a wide range of topics related to motorcycle racing, including Grand Prix motorcycle racing, MotoGP, and various other championships. It includes information on the history of motorcycle racing, notable riders and teams, and specific events such as the Venezuelan motorcycle Grand Prix and the British motorcycle Grand Prix. Additionally, the text provides details on the rules and regulations of motorcycle racing, including technical specifications and scoring systems. The text also covers various motorcycle racing seasons, including the 2002 MotoGP season, the 2003 FIM Motocross World Championship, and the 2021 FIM Moto3 World Championship. It includes information on the riders and teams that participated in these seasons, as well as the results of individual events. Furthermore, the text provides information on the 2023 FIM Moto2 World Championship, including the teams and riders that will participate in the season, as well as the schedule of events. It also covers the 2023 FIM Supercross World Championship, including the calendar and results of the event.",
        "instruction": "Which country did the champion of the 2021 Moto3 World Championship come from?",
        "type": "multi-hop"
    },
    {
        "summary": "The provided text is a collection of articles about various cities, towns, and villages in Argentina, Chile, Bolivia, and other countries. The articles provide information about the geography, climate, population, economy, and attractions of each location. Many of the articles are about small towns and villages in rural areas, often with a focus on their natural surroundings, such as mountains, lakes, and rivers. Some articles also mention the local economy, including agriculture, mining, and tourism. Some notable locations mentioned in the articles include Villa La Angostura, a town in the Andes mountains known for its natural beauty and outdoor recreational opportunities; Uspallata, a village in the Andes mountains with a rich mining history; and Puerto Fuy, a village in Chile with a scenic lake and mountain views. The articles also provide information about the history and culture of each location, including the indigenous peoples who originally inhabited the area and the European settlers who arrived later. The articles often mention the local cuisine, festivals, and traditions of each location.",
        "instruction": "Where exactly is the village known for its mining history located?",
        "type": "multi-hop"
    },
    {
        "summary": "The provided text consists of biographies of various individuals from diverse fields, including engineering, mathematics, computer science, medicine, and economics. The individuals are primarily of Chinese or Taiwanese descent, and many have made significant contributions to their respective fields. The biographies cover the individuals' early lives, education, and careers, highlighting their notable achievements and awards. Some of the notable individuals include Y. Lawrence Yao, a mechanical engineer who developed process synthesis methodology for laser forming; Jin-Yi Cai, a mathematician and computer scientist who made significant contributions to theoretical computer science; and An Wang, a Chinese-American computer engineer and inventor who co-founded Wang Laboratories. Other notable individuals include Professor Benjamin Wan-Sang Wah, a computer scientist and former provost of the Chinese University of Hong Kong; Yuan Chang, a Taiwanese-American virologist who co-discovered the Kaposi's sarcoma-associated herpesvirus; and Yingyao Hu, a Chinese-American economist who has made significant contributions to micro-econometrics and empirical industrial organization.",
        "instruction": "What research achievements did the Chinese scientist who founded Wang Laboratories accomplish?",
        "type": "multi-hop"
    },
]

PROMPTS["reasoning_instruction_demonstrations_facts"] = [
    {
        "facts": "1. Former USC athletic director Pat Haden and his family earned $2.4 million from a charitable foundation.\n2. Haden, his daughter, and sister-in-law had part-time roles with the George Henry Mayr Foundation.\n3. The foundation's contributions fell to all-time lows, with $645,000 awarded in 2014.\n4. Rubio billed the party for over $100,000 in expenses during his two years as House speaker.\n5. Rubio charged the party for repairs to his minivan, grocery bills, and plane tickets.\n6. Outgoing Senator Richard Burr received over $300,000 from donors tied to the fossil fuel industry.\n7. The donations were made over the past four years as lawmakers debated hydraulic fracturing.\n8. Senator Larry Craig was fined $3,000 and given one year's probation for disorderly conduct.\n9. Craig's arrest occurred in a public restroom at Minneapolis-St. Paul International Airport.\n10. Senator Jim DeMint received over $100,000 in donations from the banking industry in 2010.",
        "instruction": "Did Rubio reimburse more money than Haden received from the charitable foundation?",
        "type": "numerical"
    },
    {
        "facts": "1. McCool, Mississippi, is a town in Attala County with a population of 135 according to the 2010 census.\n2. The town was named for James F. McCool, Chancellor of the 6th Chancery court district of Mississippi.\n3. McCool post office was established on September 11, 1883, with Charles W. Thompson as first postmaster.\n4. Acme Township is a civil township of Grand Traverse County in Michigan, with a population of 4,375 in 2010.\n5. The township takes its name from the Greek word \"acme\" meaning summit.\n6. The population of Washington Township, Northampton County, Pennsylvania, was 5,122 at the 2010 census.\n7. Washington Township surrounds the borough cluster of Bangor and Roseto.\n8. The median income for a household in Washington Township was $48,728, and the median income for a family was $54,601.\n9. Potter Township, Beaver County, Pennsylvania, has a population of 548 and is part of the Pittsburgh metropolitan area.\n10. Traverse County, Minnesota, has a population of 3,558, making it the least-populous county in Minnesota.",
        "instruction": "Which area has the largest population: Acme Township, McCool, or Potter Township?",
        "type": "numerical"
    },
    {
        "facts": "1. Stanley Walter St Pier was an English footballer and scout, playing for West Ham United from 1929 to 1932.\n2. Darryl Westlake is a footballer who played for Walsall, Sheffield United, Kilmarnock, and Stourbridge as a defender.\n3. Tom Revill was an English cricketer and footballer, playing for Derbyshire and Stoke between 1911 and 1920.\n4. Isaac Vassell is an English professional footballer playing as a striker for Cardiff City, with 14 goals in 53 appearances for Luton Town.\n5. William Moore was an Ireland international footballer, playing for Glentoran, Ards, Falkirk, and Lincoln City between 1919 and 1926.\n6. Harry Hodgkinson was an English footballer, one of Port Vale's first players, featuring in their 1885 cup wins.\n7. Wilf Gillow was an English professional football player and manager, playing for Blackpool and managing Middlesbrough.\n8. Findlay Weir was a Scottish footballer playing for The Wednesday and Tottenham Hotspur from 1909 to 1918.\n9. Arthur Dorrell was an English international footballer, playing on the left-wing for Aston Villa between 1919 and 1931.\n10. Wally St Pier discovered several notable footballers, including John Lyall, Bobby Moore, and Geoff Hurst.",
        "instruction": "How much longer did Arthur Dorrell serve compared to Stanley Walter St Pier?",
        "type": "numerical"
    },
    {
        "facts": "1. Gmina Zwierzyn is a rural district in Strzelce-Drezdenko County, Lubusz Voivodeship, western Poland, covering 118.23 km².\n2. Slezské Rudoltice has a Renaissance chateau, a cultural center in Silesia, visited by Voltaire and Frederick II.\n3. Březnice is a village in Zlín District, Zlín Region, Czech Republic, with 1,195 inhabitants as of 2006.\n4. Stolin District, Brest Region, Belarus, has a population of 89,000 and borders Ukraine to the south.\n5. Meziboří is a town in Most District, Ústí nad Labem Region, Czech Republic, with 7,549 inhabitants as of 2020.\n6. Jedovnice is a village in Blansko District, South Moravian Region, Czech Republic, with 2,859 inhabitants as of 2020.\n7. Gmina Niebylec is a rural district in Strzyżów County, Subcarpathian Voivodeship, south-eastern Poland, covering 104.37 km².\n8. Včelince is a village in Rimavská Sobota District, Banská Bystrica Region, southern Slovakia, with a population of 236.\n9. Strachów is a village in Strzelin County, Lower Silesian Voivodeship, south-western Poland, with 150 inhabitants as of 2006.\n10. Brodek u Prostějova is a market town in Prostějov District, Olomouc Region, Czech Republic, with 1,492 inhabitants as of 2007.",
        "instruction": "Which villages have populations exceeding 500?",
        "type": "numerical"
    },
    {
        "facts": "1. Carpets in humid areas are more likely to harbor mold, which can lead to respiratory issues due to dust and material.\n2. Dust and the material from which a carpet is made contribute to mold growth and human exposure through walking.\n3. The Ohio State University study found that carpets with higher dust concentrations are more likely to grow mold.\n4. Fungi can burrow into the fibers of natural material rugs like wool, unlike synthetic fibers.\n5. The U.S. Environmental Protection Agency recommends maintaining 30-50% humidity to prevent mold growth.\n6. The Louisiana Office of Public Health found no increase in respiratory illnesses after Hurricane Katrina, but chronic symptoms worsened.\n7. University of Louisville researchers linked silica exposure to rapid lung cancer progression and found a potential target for treatment.\n8. Carbon nanotubes can be as harmful as asbestos if inhaled in sufficient quantities, causing lung damage and cancer.\n9. A University of Texas study found that workers involved in cleaning up the Deepwater Horizon oil spill were 60% more likely to develop asthma.\n10. More than one-third of Americans report health problems when exposed to common fragranced consumer products.",
        "instruction": "What factors can lead to mold growth, and how can it be prevented?",
        "type": "logical"
    },
    {
        "facts": "1. UIScrollView lags on iPhone 4 due to complex layout and large number of subviews.\n2. Using imageWithContentsOfFile instead of imageNamed improves performance in UIScrollView.\n3. UIScrollView's setContentOffset can be used to scroll to a specific point programmatically.\n4. Custom delegates can be used to achieve infinite scrolling in UICollectionView.\n5. DayFlow library can be used to implement infinite scrolling with sections in UICollectionView.\n6. scrollViewDidEndDecelerating can be used to detect when a user stops scrolling.\n7. A UIScrollView can be created programmatically with a custom frame and content size.\n8. The addSubview method is used to add a subview to a UIScrollView.\n9. A UIScrollView's content size can be set using the setContentSize method.\n10. The scrollRectToVisible method can be used to scroll a UIScrollView to a specific rectangle.",
        "instruction": "How to set scrolling for UIScrollView?",
        "type": "logical"
    },
    {
        "facts": "1. XML Schema Definition (XSD) describes structure and constraints of XML documents, aiding in data validation and interoperability.\n2. JAXB allows Java developers to access, process, and validate XML data, simplifying XML handling and reducing errors.\n3. XMLLITE is a lightweight, C++-based XML parser suitable for simple XML requirements, lacking features found in MSXML.\n4. Java class generation from XML schema is facilitated by JAXB, enabling developers to access XML data as Java objects.\n5. XML serialization and deserialization in Java involves converting Java objects to XML and vice versa, aided by libraries like JAXB.\n6. SimpleXML in PHP provides an easy-to-use interface for parsing and manipulating XML documents, with features like XPath support.\n7. XML parsing can be done using DOM (Document Object Model) or SAX (Simple API for XML), each having its strengths and use cases.\n8. XSLT (Extensible Stylesheet Language Transformations) is a language for transforming and manipulating XML documents into various formats.\n9. XML documents can be generated in Java using various libraries, including JAXB, XMLEncoder, and JDOM, each with its own strengths.\n10. XML validation involves checking XML documents against a schema or DTD to ensure they conform to a specific structure and syntax.",
        "instruction": "Can XML interact with other programming languages?",
        "type": "logical"
    },
    {
        "facts": "1. In a system with only conservative forces, mechanical energy is conserved, while in an isolated system, total energy is conserved.\n2. Newton's laws imply conservation of momentum but not energy, which is a separate principle.\n3. A system is isolated if it doesn't interact with its surroundings, meaning its total energy and mass remain constant.\n4. Isolated systems can have non-conservative forces, and mechanical energy may not be conserved due to conversion to other forms.\n5. The work-energy principle states that the net work done on an object equals its change in kinetic energy.\n6. Friction work is negative and can cause energy to be dissipated as heat, reducing mechanical energy.\n7. Conservation of energy is a fundamental principle independent of Newton's laws and applies to various physical systems.\n8. Energy is conserved in quantum mechanics, but the laws of motion differ significantly from classical mechanics.\n9. Newton's laws can be used to derive the conservation of energy and momentum principles under certain conditions.\n10. The definition of an isolated system implies that its total energy remains constant, but mechanical energy may not be conserved.",
        "instruction": "What happens to the mechanical energy of a system if it doesn’t interact with the external environment?",
        "type": "logical"
    },
    {
        "facts": "1. Josephine Meeker was a teacher and physician at the White River Indian Agency in Colorado Territory, where her father was the US agent.\n2. In 1879, Josephine's father and 10 employees were killed in a Ute attack, known as the Meeker Massacre.\n3. Josephine, her mother, and others were taken captive and held hostage by the Ute tribe for 23 days.\n4. Josephine Meeker recounted her experiences at a public hearing and provided keen insight into life as an Indian captive.\n5. She was the last celebrated white captive of Native Americans and worked in Washington, DC, and Colorado before dying young.\n6. Hubert Lacroix was an American soldier, trader, and politician who fought in the War of 1812.\n7. Lacroix was captured by the British but saved by his friend, Shawnee chief Tecumseh, and later served in the Michigan Territorial Council.\n8. Benjamin Rush Milam was a colonist, military leader, and hero of the Texas Revolution, persuading weary Texians not to back down.\n9. Milam was killed in action leading an assault into San Antonio, which eventually resulted in the Mexican Army's surrender.\n10. The Battle of Pease River occurred on December 18, 1860, where a group of Comanche Indians were killed by Texas Rangers and militia under Captain Sul Ross.",
        "instruction": "Which event happened first, the Battle of Pease River or the Meeker Massacre?",
        "type": "temporal"
    },
    {
        "facts": "1. The 325,000-year-old site in Armenia shows that human technological innovation occurred intermittently, not spreading from a single point of origin.\n2. Analysis of artifacts suggests that humans in Armenia used both biface and Levallois technologies simultaneously, 325,000 years ago.\n3. A study of tools from the Pech Merle cave in southern France used indigenous tracking expertise to identify five individuals and their age and sex.\n4. Neanderthals used tar made from tree bark as an adhesive to craft weapons and other tools, 200,000 years ago.\n5. Researchers discovered three possible methods Neanderthals used to extract tar from birch bark, using only available tools and materials.\n6. A study of stone tools from Sri Lanka found evidence of microliths, small stone tools, used for hunting in tropical rainforests 48,000 years ago.\n7. The human ability to teach and use complex tools may have evolved together, with teaching becoming advantageous as tools became more complex.\n8. A study of Hadza hunters found that they could manufacture and transmit bow-making technology with only partial causal knowledge, without needing experts.\n9. The oldest ground-edge stone tool in the world, found in Northern Australia, is dated to 35,000 years ago, pre-dating previous examples.\n10. Analysis of artifacts from the Mughr el-Hamamah site in Jordan found evidence of complex fishing technology, including the use of artificial lures, 12,000 years ago.",
        "instruction": "Did ancient people use small stone tools or artificial lures first?",
        "type": "temporal"
    },
    {
        "facts": "1. Aranyakam is a 1988 Malayalam film directed by Hariharan, written by M. T. Vasudevan Nair, starring Saleema, Devan, and Vineeth.\n2. Aayiram Roobai is a 1964 Tamil film directed by K. S. Gopalakrishnan, starring Gemini Ganesan, Savitri, Nagesh, and M. R. Radha.\n3. Raksha Rekha is a 1949 Telugu swashbuckling adventure fantasy film directed by R. Padmanabhan, starring Akkineni Nageswara Rao and Bhanumathi Ramakrishna.\n4. Kanaka Simhasanam is a 2006 Malayalam comedy film directed by Rajasenan, starring Jayaram, Karthika, and Lakshmi Gopalaswamy.\n5. Jai Sriram is a 2013 Telugu action film directed by Balaji N. Sai, starring Uday Kiran and Reshma Rathore.\n6. Vaazha Vaitha Deivam is a 1959 Tamil romantic drama film directed by M. A. Thirumugam, starring Gemini Ganesan and B. Sarojadevi.\n7. Inspector Vikram is a 1989 Kannada film directed by Dinesh Baboo, starring Shivarajkumar and Kavya in the lead roles.\n8. Loktak Lairembee is a 2016 Manipur film directed and produced by Haobam Paban Kumar, starring Ningthoujam Sanatomba and Sagolsem Thambalsang.\n9. Radha Gopalam is a 2005 Telugu film directed by Bapu, starring Srikanth and Sneha, based on the Hollywood movie Adam's Rib.\n10. Chambal is a 2019 Indian Kannada thriller film directed by Jacob Verghese, starring Satish Ninasam and Sonu Gowda in the lead roles.",
        "instruction": "Which film was released first, Loktak Lairembee or Aayiram Roobai?",
        "type": "temporal"
    },
    {
        "facts": "1. Mike Weir is a Canadian golfer born in 1970, winner of the 2003 Masters Tournament.\n2. Michael McGoldrick is an Irish flute and tin whistle player born in 1971.\n3. Michael Gallagher is an American politician, historian, and author, born in various years.\n4. Mark Lee may refer to several individuals, including actors, musicians, and politicians.\n5. Michael Chiang is a Singaporean playwright and screenwriter born in 1955.\n6. After Marcuse is an Australian TV film starring Diane Craig and Penne Hackforth-Jones.\n7. Greengard is a surname shared by several notable individuals, including Irene, Leslie, and Michael.\n8. Mark Lee may refer to several individuals, including athletes, musicians, and politicians.\n9. Michael Wallace may refer to several individuals, including politicians, athletes, and a lawyer.\n10. Michael Chack is an American former competitive figure skater born in 1971.",
        "instruction": "Which athletes were born after 1960?",
        "type": "temporal"
    },
    {
        "facts": "1. The Hakim family, prominent Shiite Islam scholars, claim descent from the Prophet Muhammad, with a history in Iraq.\n2. Grand Ayatollah Muhsin al-Hakim was a key figure in defending Islam and Muslims, becoming the sole Marja' in 1961.\n3. Al-Hakim's son Abdul Aziz al-Hakim led the Islamic Supreme Council of Iraq, the largest political party in Iraq.\n4. Al-Tabarani, a renowned hadith scholar, traveled extensively, collecting and writing various Hadith books, including Al-Mu'jam Al-Kabir.\n5. Al-Tabarani is known for his three primary works on hadith, excluding traditions of Abu Hurayra in Al-Muʿjam al-Kabīr.\n6. Ahmed ibn Abi Mahalli led a revolt against the Saadi Sultan Zaydan Bin Ahmed in Morocco, proclaiming himself mahdi.\n7. Sabat M. Islambouli was one of the first Syrian female physicians, graduating from the Woman's Medical College of Pennsylvania in 1890.\n8. Abdallah al-Ghalib Billah succeeded his father Mohammed ash-Sheikh as Sultan of Morocco, reigning from 1557 to 1574.\n9. Mukarrib, a title in ancient South Arabia, refers to a 'federator' or 'priest-king,' possibly indicating the ruler of a group of tribes.\n10. Al-Hakam ibn Abi al-As, the father of Marwan I, was a staunch opponent of Muhammad, later pardoned and allowed to return to Mecca.",
        "instruction": "Who was the son of the Marja’ in 1961?",
        "type": "multi_hop"
    },
    {
        "facts": "1. Ambrosia beetles are a type of insect that eat ambrosia fungus and are involved in research on taxonomy and diversity.\n2. The Spartans named new species of ambrosia beetles after iconic female science fiction characters, including Nyota Uhura and Kara Thrace.\n3. The ambrosia beetles are thought to have originated 20 million years ago in Southeast Asia and emigrated across the tropics.\n4. Aer-ki Jyr\'s novel \"Apex\" features a human who awakens from cryosleep and tries to reclaim humanity\'s dominance in the galaxy.\n5. \"Midsummer Century\" by James Blish is a novella about a man whose mind is transported to the far future and becomes an oracle.\n6. The book \"Beyond\" by Chris Impey discusses the history of space travel and the future trajectory of human exploration.\n7. The novel \"The Stars, Like Dust\" by Isaac Asimov features a rebellion against the Tyranni, a powerful empire that rules over 50 planets.\n8. \"The Collapsing Empire\" by John Scalzi is a science fiction novel that explores a future where humanity has colonized other planets and formed an interstellar empire.\n9. \"Children of Time\" by Adrian Tchaikovsky is a novel about a group of humans who flee a dying Earth and discover a planet where a scientific experiment has created intelligent spiders.\n10. \"The Helix and the Sword\" by John McLoughlin is a science fiction novel set in a future where humanity has abandoned Earth and formed tribes in space, with a focus on a biological computer called Pantalog 5.",
        "instruction": "What is the diet of the species that migrated to the tropics as mentioned in the material?",
        "type": "multi_hop"
    },
    {
        "facts": "1. Carriers of a GALNT2 gene mutation are better at clearing triglycerides from their systems due to altered enzyme interactions.\n2. Lipid metabolism plays a crucial role in various diseases, including diabetes, metabolic syndromes, and certain cancers.\n3. The human body contains thousands of types of fats, or lipids, essential for energy storage, hormone production, and cell membrane structure.\n4. Researchers have identified a molecular switch, TBL1, that is reduced in people with fatty liver disease, which may explain disease development.\n5. A team of scientists has discovered that the organization of the human genome relies on physics of different states of matter.\n6. The genome's genetic information is encoded in the DNA molecule, requiring proper reading and processing for human health and aging.\n7. Researchers have catalogued more than 20,000 brain cells in one region of the mouse hypothalamus, revealing new cell types and targets for obesity treatment.\n8. A study has identified hundreds of proteins that may contribute to the onset of common metabolic diseases such as type 2 diabetes.\n9. Phospholipids, a type of fat molecule, may play a significant role in autoimmune diseases, including psoriasis, contact hypersensitivities, and allergies.\n10. Scientists have developed a powerful tool for exploring and determining the inherent biological differences between individuals, paving the way for precision medicine.",
        "instruction": "Which substance is related to energy storage and diabetes?",
        "type": "multi_hop"
    },
    {
        "facts": "1. Ōura Station in Niigata was a train station on the Yahiko Line, opened in 1927 and closed in 1985.\n2. Nakagawa Station is an underground metro station in Yokohama, operated by the Yokohama Municipal Subway's Blue Line.\n3. Nakakido Station is a railway station on the Keikyu Main Line in Kanagawa-ku, Yokohama, operated by Keikyu Railway.\n4. Tara Station is a railway station in Tara, Saga Prefecture, operated by JR Kyushu and located on the Nagasaki Main Line.\n5. The Osaka Loop Line is a railway loop line in Japan, operated by JR West, encircling central Osaka.\n6. Itaya Station is a railway station on the Ōu Main Line in Yonezawa, Yamagata Prefecture, operated by East Japan Railway Company.\n7. Kita-Chigasaki Station is a train station in Chigasaki, Kanagawa Prefecture, operated by East Japan Railway Company.\n8. Naruko-Onsen Station is a railway station on the Rikuu East Line in Ōsaki, Miyagi Prefecture, operated by East Japan Railway Company.\n9. The Yamaman Yūkarigaoka Line is a people mover line in Sakura, Chiba, operated by Yamaman Co., Ltd.\n10. Tategahana Station is a railway station on the Iiyama Line, East Japan Railway Company, in Toyono-Kanisawa, Nagano Prefecture.",
        "instruction": "What other train stations are operated by the railway company running Naruko-Onsen Station?",
        "type": "multi_hop"
    },
]

PROMPTS["synthesize_response_instruction_start"] = """
{instruction}

{context}
"""

PROMPTS["synthesize_response_instruction_end"] = """
{context}

{instruction}
"""

# PROMPTS["reference_based_evaluation"] = """
# [Instruction]
# Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. In addition to the user question, you are also given a reference response. This is the best possible response provided by a human expert. You should evaluate the assistant’s response based on this. A good assistant’s response should have the same answer as the reference response. Begin your evaluation by providing a short explanation. Be as objective as possible. 
# After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format:
# "[[rating]]", for example: "Rating: [[5]]".

# [Question]
# {question}

# [Reference Response]
# {reference}

# [The Start of Assistant’s Response]
# {response}
# [The End of Assistant’s Response]
# """

PROMPTS["reference_based_evaluation"] = """
Here is a question along with two responses: one is the reference response, and the other is the predicted response. Please determine whether the two responses provide the same answer to the question. Respond with “True” or “False” directly.

[Question]
{question}

[Reference Response]
{reference}

[Predicted Response]
{prediction}
"""

PROMPTS["reference_free_evaluation"] = """
[Context]
{context}

[Question]
{question}

[Predicted Response]
{prediction}

Please evaluate the correctness of the predicted response based on the context and the question. Begin your evaluation by providing a brief explanation. Be as objective as possible. After giving your explanation, you must rate the response on a scale from 1 to 5, following this format exactly: “[[rating]]”. For example, “Rating: [[3]]”.
"""
