def add(num1, num2):
    return num1 + num2

def substract(num1,num2):
    return num1 - num2

def multiply(num1,num2):
    return num1 * num2

def divide(num1,num2):
    return num1 * num2

operation={
    '+':add,"-":substract,'x':multiply,'/':divide
    }

a=int(input("Enter a first number:- "))
b=int(input("Enter a second number:- "))

for keys, value in operation.items():
   print(keys)

c=input("Pick the symbol you want:- ")
operation_function=operation[c]
answer=operation_function(a,b)

print(f"Your answer is:-  {answer}")

