class Library:
    def __init__(self,listofbooks):
        self.books=listofbooks

    def displayAvailableBooks(self):
        print(f"Books present in this library are: ")
        for book in self.books:
            print("\t*" + book)
    def borrowBook(self,bookname):
        if bookname in self.books:
            print(f"You have been issued {bookname}. Please keep it safe")
            self.books.remove(bookname)
            return True
        else:
            print("Sorry this book is not available")
            return False
    def returnBook(self,bookname):
        self.books.append(bookname)
        print("Thanks for returning this book!")

class Student:
    
    def requestBook(self):
        self.book=input("Enter the book you wants to borrow: ")
        return self.book
    
    def returnBook(self):
        self.book=input("Enter the book you wants to return: ")
        return self.book

if __name__=="__main__":
    l=Library(["Algorithms","Django","Clrs","python notes"])
    # l.displayAvailableBooks()
    student=Student()
    while(True):
        welcomemsg= '''
        =====WELCOME TO CENTRAL LIBRARY=======
        Please choose an option:
        1. Listing all the books
        2. Request a book
        3. Return a book
        4. Exit a library'''
        print(welcomemsg)
        
        a=int(input("Enter a choice "))
        if a==1:
            l.displayAvailableBooks()
        elif a==2:
            l.borrowBook(student.requestBook())
        elif a==3:
            l.returnBook(student.returnBook())
        elif a==4:
            print("Thanks for using this library")
            exit()
        else:
            print("Invalid choice!")
            
        

        