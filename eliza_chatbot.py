# Natural Language Toolkit: Eliza
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Steven Bird <stevenbird1@gmail.com>
#          Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

# Based on an Eliza implementation by Joe Strout <joe@strout.net>,
# Jeff Epler <jepler@inetnebr.com> and Jez Higgins <mailto:jez@jezuk.co.uk>.

# a translation table used to convert things you say into things the
# computer says back, e.g. "I am" --> "you are"

from nltk.chat.util import Chat, reflections

# a table of response pairs, where each pair consists of a
# regular expression, and a list of possible responses,
# with group-macros labelled as %1, %2.

pairs = (
    (
        " (.*)",
        (
            "Aap ko %1 kiu chahiye?",
            "Kia ye waqie apki madad krega agr apko %1 ye miljae?",
            "Kia apko yaqeen hai ke ap ko %1 chahiye?",
        ),
    ),
    (
        r"Tum kiu (.*) nahi kar rahi?",
        (
            "Kia aap waqie sochte hain ke main %1 nahi kar rahi hun?",
            "Shyd bil aakhir main %1 krlun.",
            "Kia ap waqie chahtay hain ke main %1 kru?",
        ),
    ),
    (
        r"Main kiu (.*) nahi kar sakta?",
        (
            "Kia aap ko lagta hai ke aap %1 kar sakenge?",
            "Agar aap %1 kar saktay, to aap kia kartay?",
            "Mujeh nahi pata, aap kiu %1 nahi krrhe?",
            "Kia aapne waqie koshish ki hai?",
        ),
    ),
    (
        r"Main (.*) nahi kar sakta",
        (
            "Aap ko kese pata ke aap %1 nahi kar saktay?",
            "Zahiran aap %1 kar saktay agar aap koshish kartay.",
        ),
    ),
    (
        r"Main (.*) hun",
        (
            "Kia aap mere pas ae hain kiunke aap %1 hain?",
            "Aap kab se %1 hain?",
            "Aap ko %1 hote hue kesa mehsoos horha hai?",
        ),
    ),
    (
        r"Main (.*) karta hun",
        (
            "Aap ko %1 hona kesa lagta hai?",
            "Kia aap %1 honay enjoy krtay hain?",
        ),
    ),
    (
        r"Kia aap (.*) hain?",
        (
            "Isse kiu farq prta hai ke main %1 hun ya nahi?",
            "Kia ye behtar hota agar main %1 hoti?",
        ),
    ),
    (
        r"Kia (.*)",
        (
            "Aap ye kiu puchre hain?",
            "Iska jawab apki kis tarhan madad karega",
            "Aap kia sochte hain is baray main?",
        ),
    ),
    (
        r"Kese (.*)",
        (
            "Aap kia samjhtay hain is baray main?",
            "Ghaliban ap apnay sawal ka khud hi jawab deskte hain."
        ),
    ),
    (
        r"Kiunke (.*)",
        (
            "Kia yehi asal wajah hai?",
            "Kia koi aur wajah apke zehn main aarahi hai?",
            "Kia ye wajah kisi or msle kesath bhi hai?",
            "Agar %1 hai to mazeed kia kia hoga?",
        ),
    ),
    (
        r"(.*) mazrat (.*)",
        (
            "Basa Auqaat apki mazrat hi darkaar hoti hai",
            "Apki jazbat kis tarha hote hain jab aap mazrat krte hain?",
        ),
    ),
    (
        r"Salam (.*)",
        (
            "Waalaikum Asalam, Main bohat khush hun ke aap aaj agaye",
        ),
    ),
    (
        r"Mujhe lagta hai (.*)",
        ("Kia apko %1 main koi shak hai?", "Kia aapko waqie ye lagta hai?"),
    ),
    (
        r"(.*) dost (.*)",
        (
            "Apne dosto ke baaray main mazeed bataiye",
            "Jab aap dosto ke baray main sochte hain to apke zehn mn kia aata hai?",
        ),
    ),
    (
        r"Kia aap (.*)?",
        (
            "Aap kiu sochte hain ke main %1 nahi kar sakti?",
            "Aap kiu puchrhe hain ke main %1 kar sakti hu ya nahi?",
        ),
    ),
    (
        r"Kia Main (.*)\?",
        (
            "To zahiran aap %1 nahi chatay?",
            "Kia aap waqian %1 krna chahtay hain?",
        ),
    ),
    (
        r"Kia Kaheen (.*)",
        (
            "Kia aapko lgta hai ke %1 hai?",
            "Ji, ye mumkin hai ke %1 ho",
        ),
    ),
    (
        r"Main (.*)",
        (
            "Main samjh rahi hu",
            "Aap kiu kehrhe hain ke aap %1 hain?",
        ),
    ),
    (
        r"Aap (.*)",
        (
            "Hamein apke baray main baat karni chahiye, mere baray main nahi",
            "Aap mere baray main ye kiu kehrhe hain?",
            "Aap ko kia farq par skta hain mere %1 hone se?",
        ),
    ),
    (
        r"Kiu (.*)",
        (
            "Aap mujhe is baray main mazeed bataenge?",
            "Aap kiu sochte hain %1"
        )
    ),
    (
        r"Main chahta hu (.*)",
        (
            "Kia hoga agar apko %1 mil jaye?",
            "Aap %1 kiu chahtay hain?",
        ),
    ),
    (
        r"(.*) walida(.*)",
        (
            "Apni walida ke baray main mazeed bataiye",
            "Aapki walida apko kesa mehsoos krwatay thay?",
            "Apko apnai walida ke baray main kia lgta hai?",
            "Kia aapka apni waida se taaluq apki aaj ke ahsaasat ki wajah hai?",
        ),
    ),
    (
        r"(.*) walid(.*)",
        (
            "Apne walid ke baray main mazeed bataiye",
            "Aapke walid apko kesa mehsoos krwatay thay?",
            "Apko apnay walid ke baray main kia lgta hai?",
            "Kia aapka apne waid se taaluq apki aaj ke ahsaasat ki wajah hai?",
        ),
    ),
    (
        r"(.*) bachpan(.*)",
        (
            "Kia aap ke bachpan mn qareebi dost the?",
            "Aapke bachpan ki koi dilchaspt yaadein?",
            "Kia aap ko koi bachpan ka acha ya khofnak khuwab yaad hai?",
            "Kia dusre bachay apko kabhi kabhar tang kia kartay thay?",
        ),
    ),
    (
        r"(.*)\?",
        (
            "Aap ye kiu puchrhe hain?",
            "Aap sochie kia ap apne sawal ka jawab deskte hain?",
            "Ghaliban jawab ap ke andar hi chupa hai",
            "Aap mujhe kiu nahi bata dete?",
        ),
    ),
    (
        r"quit",
        (
            "Mujhse guftugu farmanay ka shukria",
            "Khuda Hafez",
            "Shukria, ye 2000 rs. honge",
        ),
    ),
    (
        r"(.*)",
        (
            "Baraye mehrbani, mazeed btayie",
            "Chlen kuch aur baat karte hain, apni family ke baray main kuch batayie",
            "Kia iski wazahat kar sakte hain",
            "Aap %1 kese kehrhe hain?",
            "Sahi Sahi",
            "Ye dilchasp baat hai",
            "%1.",
            "Acha. Aur isse aap kia andaza lagatay hain?",
        ),
    ),
)

eliza_chatbot = Chat(pairs, reflections)


def eliza_chat():
    print("Therapist\n---------")
    print("Program se plain Roman main baat kijiye, aam upper-case, lower-case-")
    print('letters or punctuations kesath. \nJab aap quit krna chahen to "khatam" type kijiye')
    print("=" * 72)
    print("Salam. Aap aaj kesa mehsoos krrhe hain?")

    eliza_chatbot.converse()


def demo():
    eliza_chat()


if __name__ == "__main__":
    demo()
