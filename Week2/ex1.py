#Slicing

a = "Applied Machine Learning"
print(a[:7])
print(a[8:])
print(a[-7:])

#lists

mylist = ["apple", "banana", "cherry"]
print(mylist)
mylist.append("apple")
print(mylist)

#sets

myset = {"apple", "banana", "cherry"}
print(myset)
myset.add("apple")
print(myset)

#dictionary

mydict = {
   "brand": "Ford",
   "model": "Mustang",
   "year": 1964
}
print(mydict)
print(mydict['brand'])
mydict['brand'] = 'GM'
print(mydict)

#if… else

a = 33
b = 33
if b > a:
    print("b is greater than a")
elif a > b:
    print("a is greater than b")
else:
    print("a and b are equal")

print("a is greater than b") if a > b else print("a is not greater than b")

print("a is greater than b") if a > b else print("a and b are equal") if a == b else print("b is greater than a")

#while loop

i = 1
while i < 4:
   print(i)
   i += 1

#for loop

fruits = ["apple", "banana", "cherry"]
for x in fruits:
   print(x)

#lambda

x = lambda a, b : a * b
print(x(5, 6))

#datetime

import datetime
x = datetime.datetime.now()
print(x.year)

print(x.strftime("%m/%d/%Y"))