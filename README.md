# Computing and Numerics - Assignment A2 [40 marks]

Solve the exercises in the notebook `A2.ipynb`. Please read the guidance below carefully.

---

### Question 3.4: new initial guesses

Please ignore the initial guesses given in the last code cell for question **3.4**, and instead use the following:

```python
# Initial conditions
uv0 = np.array([[0.9, 0.6], [0., 0.4], [-1.7, -0.3]])
```

---

### Presentation, plots, and code comments

- Your code should be **well-commented**, with a comment explaining each step of your code. This is to test your understanding of the code you have written.
- When printing any results, they should be displayed in a clear and easily readable manner. For instance, avoid printing values or arrays on their own, without any context.
- Your code must generate and display all relevant output when run, and all figure formatting must be performed programmatically and not via the interactive plotting interface. Rerun your code cells after editing your code, to make sure that the output is updated.
- Your discussions in Markdown cells should use appropriate Markdown formatting when necessary.

**Up to half** of the marks for a given question may be deducted for a missing, incomplete, or inaccurate code commentary, poorly displayed results, or poorly labelled or unclear plots.

---

### Working on your project with git and GitHub

- You will submit the assignment through Gradescope, by providing a link to your GitHub repository, exactly the same way you submitted A1.
- While working on the project, **commit your changes often** -- every time you make progress on a subtask. If you tend to forget to do this regularly, you could e.g. set a timer to remind you to commit every hour or so.
- You don't necessarily need to push your changes to GitHub every time you commit, although we'd strongly recommend that you **push them regularly**. This ensures that
    - you won't have any last-minute technical issues when it comes to submitting, and
    - you have an **online backup** of your work, just in case e.g. your computer breaks down.

---

### Academic integrity

This is an **individual** assignment -- just like for the Coderunner quizzes and for A1, the work your submit must be your own, to reflect and assess your own understanding and knowledge.

#### Collaboration vs. collusion

Collaboration is fine, but collusion is not. Concretely, this means that discussing the assignment **in broad terms** with others students is fine (and encouraged), as well as giving each other hints or general advice on how to approach a problem. You can use Piazza for this, for example -- if you are stuck, then please ask for help! However, you are **not permitted** to share your working (even partially) with other students -- that includes your code, any detailed description or explanation of code, and any results or analysis you perform.

For example:
- Alice and Bob are discussing the assignment on a Zoom call. Bob's code is not working for one of the questions, and he can't figure out why. He asks Alice how she's tackled the problem, and she explains her approach in broad terms. This gives Bob an idea, and he tries it later. *This is all fine.*
- Bob's idea doesn't work out, and he calls Alice again. He shares his screen with her to show his code. *This is getting dangerous* -- here's why:
    - Alice helps him with understanding the error, and gives him some pointers and ideas to try, without explaining the problem or the solution in much detail. *That would still be fine.*
    - Alice is stuck on the next question, though, and spots a couple of lines of Bob's code at the bottom of the screen. She uses some of that code for the next question in her submission. This is not OK: *both Bob and Alice have now committed misconduct* -- Alice by using Bob's code, and Bob by sharing his screen.
- Bob is still stuck. He posts his code for that question on Piazza. Some students help and also give him some  general advice. Charlie sees the post on Piazza, and didn't know how to start that question. Charlie uses some of Bob's code, with some corrections to fix the problems, and submits it for the assignment. *This is also misconduct* by both Bob and Charlie.
- Bob is still stuck (poor Bob!). It's getting very close to the deadline now, so he asks his friend Dara to *pleaaaase* show their solution, he promises not to copy it. Bob and Dara are really good friends, so Dara finds it difficult to refuse and sends their code. Bob rewrites Dara's code by changing some variable names, rearranging a bit, and paraphrasing the code comments so that they are "in his own words". *This is misconduct* by both Bob and Dara.

Use and trust your own judgement. It's important to understand that even with the best intentions, you expose yourself to academic misconduct as soon as you show your code to another student.

#### Providing references

Most of the code in your submission must be **authored by you**. That being said, you may use any code from the course material (e.g. workshop tasks, tutorial sheets, videos), without citing it.

You may also use **small pieces of code** (a few lines max at a time) that you found elsewhere -- e.g. examples from the documentation, a textbook, forums, blogs, etc... You may use this code *verbatim* (i.e. almost exactly as you found it), or adapt it to write your own solution.

A programming assignment is just like any other academic assignment -- and therefore, **you must provide a citation for any such code**, whether you use it *verbatim* or adapt it. To do so, include a code comment at the start of your script or notebook cell, indicating:
- the line numbers where the code was used or adapted,
- the URL of the source (or, if it's from a book, a full reference to the book),
- the date you accessed the source,
- the author of the code (if the information is available).

You can use this template -- delete one of the URL or book reference lines as appropriate:
```python
# Lines X-Y: Author Name
# URL: http://...
# Book Title, year published, page number.
# Accessed on 31 Nov 2020.
```

You must also provide **detailed code comments** for any such code, in your own words, to demonstrate that you fully understand how it works -- you will lose marks if you use external code without explaining it, even if it's cited correctly.

Remember to exercise caution if you use any code from external sources -- there are a lot of blogs and forums out there with very bad code! I'd recommend that you review the Week 4 video on searching the documentation.

With all that, we trust that you'll be able to use your best judgement, and to cite your sources appropriately -- if anything is not clear, please do ask. Note that **all submissions** will be automatically checked (and manually reviewed) for plagiarism and collusion, and [the University's academic misconduct policy](https://www.ed.ac.uk/academic-services/staff/discipline/academic-misconduct) applies.
