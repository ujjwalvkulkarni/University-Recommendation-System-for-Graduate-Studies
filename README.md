# University-Recommendation-System-for-Graduate-Studies
A small step towards mitigating the ambiguity of selecting universities in the USA for Master's in Computer Science.

Shortlisting of universities according to their application profile, is a major problem faced by students pursuing their master’s degrees. The master's application packet includes quantifiable scores like the GRE score, English proficiency test score, and College grades. Apart from these scores, another important document is the Statement of Purpose. A Statement of Purpose can be the difference between an admit and a reject. There are many university recommendation applications but none of them takes the Statement of Purpose into account. This project aims to help undergraduate students overcome this ambiguity and provide a list of colleges that are safe to apply to. The Statement of Purpose is analyzed using Natural Language Processing. Gradient Boost regression technique is used in order to train and test the data set. Along with it, various functions of NLTK are used to extract features from the essay. The recommender provides a list of 8 colleges from which the user can most likely get admission from. This project will prove to be extremely useful for the student community as it provides proper guidance which is not available for free. The recommendation system is a small step in the right direction to ensure that students apply for the right universities.

Algorithm
1. The student logs into the system and enters the details.
2. The Statement of Purpose is uploaded in .csv format.
3. Essay Rater scores the Statement of Purpose based on the trained model.
4. The score is appended to the tuple of the student.
5. Every feature in the tuple is assigned a weight
according to the score.
6. The weights are summed up in order to find the
category of universities suitable for the profile.
7. The list of universities is displayed to the user.

Run the Input.py file and fill in the details to evaluate the profile. The SOP should be saved in the SOP.csv file to be evaluated correctly. Fill in the details and get a list of colleges you are most likely to get acceptance from. 

P.S. The criteria on which the admissions are rolled out is ambiguous as the admission committee does not explicitly mention it. But, based on the admission trend, we were able to infer the trend of admissions for different universities. This is just a small step towards solving the ambiguous problem of college shortlisting.

Our paper on URS won the best paper certificate in the ”National Conference on Artificial Intelligence and Intelligent Information Processing” (NCAIIIP’20) organised by department of Software Engineering, SRM IST. Link: https://github.com/ujjwalvkulkarni/University-Recommendation-System-for-Graduate-Studies/blob/main/NCAIIIP-KARAN%20%20UJJWAL.pdf

![](https://github.com/ujjwalvkulkarni/University-Recommendation-System-for-Graduate-Studies/blob/main/Conference%20certificate.png)
