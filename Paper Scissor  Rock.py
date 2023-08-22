def gamewin(comp,you):
    if comp =='s':
        if you == 's':
            return None
        elif you == 'p':
            return True
        elif you =='sc':
            return False
    elif comp =='p':
        if you=='p':
            return None
        elif you=='s':
            return False
        elif you=='sc':
            return True
    elif comp =='sc':
        if you=='sc':
            return None
        elif you=='s':
            return True
        elif you=='p':
            return False
you=input("\nYour turn! Stone(s), Paper(p), Scissor(sc):- ")           
print("Comp turn : Stone(s), Paper(p), Scissor(sc)")
import random
randno=random.randint(1,3)
if randno ==1:
    comp= 's'
elif randno==2:
    comp='p'
elif randno==3:
    comp="sc"

a=gamewin(comp,you)

print(f"\nComputer choosed= {comp}")
print(f"You choosed = {you}")

if a==None:
    print("Game is tie")
elif a==True:
    print("Congratualtions! You won the match")
elif a==False:
    print("Sorry! You lose the match")



