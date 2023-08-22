import random
letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w',
         'x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T',
         'U','V','W','X','Y','Z']
numbers=['0','1','2','3','4','5','6','7','8','9']
symbols=['@','#','*','$','!','%','&','(',')']
print("\n======= Welcome to the Pypassword gnerator! ==========\n")

st_letters=int(input("How many letters would you like in your password: "))
st_numbers=int(input("How many numbers would you like in your password: "))
st_symbols=int(input("How many symbols would you like in your password: "))
import random
password_list=[]
for char in range(st_letters):
    random_char=random.choice(letters)
    # print(random_char) for knowing what comes in the output
    password_list += random_char

for char in range(st_numbers):
    random_char=random.choice(numbers) 
    password_list += random_char 

for char in range(st_symbols):
    random_char=random.choice(symbols) 
    password_list += random_char

random.shuffle(password_list)
# print(password_list)

password=""
for char in password_list:
    password += char
    
print(f"Congratulations! Your password has been generated = {password}")


    


