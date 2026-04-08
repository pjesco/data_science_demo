print("Welcome to Data Science")

# I am learning or reviewing Python programming language

"""
This
is
a
block
comment
"""

first_name = "greg"
last_name = "reis"

print(type(first_name))
print(type(last_name))
print(first_name, last_name)
print("FIU","Miami",54000,True)
print(first_name + last_name)
print(first_name + " " + last_name)

print(len(first_name))
print(first_name[0])

a = 10
b = 3

print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a//b)
print(a**b)
print(a % b)

print(a > b)
print(a == b)
print(a != b)
print(a < b)

""" Next class we continue with strings, lists, dictionaries, functions and object oriented"""

university = "fiu "
print(type(university))
print(len(university))
print(university.title())
print(university.upper())
print(university.lower())
print(university[0])
print(university[3])

print(2*3)
print(3*"3")

## Lists

professors = ["greg","kianoosh","richard","trevor",
              "debra","patricia","mark"]
print(len(professors))
print(professors[0])
print(professors[-1])
print(professors[3:5])
print(professors[4:])
print(professors[:5])
print(professors[:])
professors.append("jason")
print(professors)
professors.extend(["leo","todd","kevin"])
print(professors)
professors.insert(2,"heather")
print(professors)
professors[3]="stephanie"
print(professors)
x=professors.pop(1)
print(x,professors)
professors.remove("jason")
print(professors)
professors.sort()
print(professors)
professors.sort(reverse=True)
print(professors)

for i in professors:
    print(i.title())
print("FIU")

## Dictionaries

waterData = {
    "date":["01/08/26", "01/10/26", "01/12/26"],
    "temperature":[79.2, 87.3, 92.4],
    "odo":[6.7, 5.6, 3.5],
    "salinity":[65, 66, 57],
}

print(type(waterData))
print(waterData.keys())
print(waterData.values())
print(waterData["temperature"])

for j,k in waterData.items():
    print(f"{j.title()}: {k}.")

import pandas as pd

df = pd.DataFrame(waterData)
print(df)