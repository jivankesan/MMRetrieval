def class_conversion(name):
    name_to_int_dict = {
            "Cook.Cleandishes": 1,
            "Cook.Cleanup": 2,
            "Cook.Cut": 3,
            "Cook.Stir": 4,
            "Cook.Usestove": 5,
            "Cutbread": 6,
            "Drink.Frombottle": 7,
            "Drink.Fromcan": 8,
            "Drink.Fromcup": 9,
            "Drink.Fromglass": 10,
            "Eat.Attable": 11,
            "Eat.Snack": 12,
            "Enter": 13,
            "Getup": 14,
            "Laydown": 15,
            "Leave": 16,
            "Makecoffee.Pourgrains": 17,
            "Makecoffee.Pourwater": 18,
            "Maketea.Boilwater": 19,
            "Maketea.Insertteabag": 20,
            "Pour.Frombottle": 21,
            "Pour.Fromcan": 22,
            "Pour.Fromkettle": 23,
            "Readbook": 24,
            "Sitdown": 25,
            "Takepills": 26,
            "Uselaptop": 27,
            "Usetablet": 28,
            "Usetelephone": 29,
            "Walk": 30,
            "WatchTV": 31
        }
        
    return name_to_int_dict.get(name, 0)

        