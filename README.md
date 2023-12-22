# Principles of Concurrency and Parallelism in Python

> There is no doubt that the grail of efficiency leads to abuse.
> Programmers waste enormous amounts of time thinking about,
> or worrying about, the speed of noncritical parts of their programs,
> and these attempts at efficiency actually have a strong negative
> impact when debugging and maintenance are considered.
> We should forget about small efficiencies, say about 97% of the time:
> premature optimization is the root of all evil.
>
> Yet we should not pass up our opportunities in that critical 3%.
> A good programmer will not be lulled into complacency by such reasoning,
> he will be wise to look carefully at the critical code;
> but only after that code has been identified.
> It is often a mistake to make a priori judgments about
> what parts of a program are really critical,
> since the universal experience of programmers who
> have been using measurement tools has been that their intuitive guesses fail.
>
> -- [Donald Knuth](https://doi.org/10.1145/356635.356640)

> I've always thought this quote has all too often led software designers into serious mistakes
> because it has been applied to a different problem domain to what was intended.
> The full version of the quote is
> "We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil."
> and I agree with this.
> Its usually not worth spending a lot of time micro-optimizing code before its obvious where the performance bottlenecks are.
> But, conversely, when designing software at a system level, performance issues should always be considered from the beginning.
> A good software developer will do this automatically, having developed a feel for where performance issues will cause problems.
> An inexperienced developer will not bother, misguidedly believing that a bit of fine tuning at a later stage will fix any problems.
>
> I've worked on systems where the architects adhered to "Hoare's Dictum".
> All goes well until realistic large-scale testing is performed and it becomes
> painfully obvious that the system is never going to scale upwards.
> Unfortunately by then it can be very difficult to fix the problem without a large amount of re-design and re-coding.
>
> -- [Charles Cook](https://web.archive.org/web/20220411104143/http://www.cookcomputing.com/blog/archives/000084.html)

While *premature* optimization can indeed lead to problems,
it is important to consider potential performance issues from the beginning of any project
in order to *avoid* the need for optimizations later.
This is particularly important when the optimizations will rely on concurrency and/or parallelism,
because the structure of the resulting code can vary dramatically depending on what kind of concurrency/parallelism technique is used.
Making the right structural choices early on (e.g., between threads, processes, or asynchronous programming) can save a lot of time and effort.

The goal of this talk is to provide a framework for thinking about how concurrency and parallelism work in Python,
and thus to empower you to make informed decisions about which tools or techniques to use in your projects.
This is *not* a talk about how to write concurrent/parallel code in Python, or indeed how to optimize code at all.
The goal is for you to be able to decide, based on the needs of any given project, which approach you should use for it.
We will discuss what concurrency and parallelism are, how they differ from each other,
how they are realized in computers in general and Python in particular,
and how to decide which approach to use for a given project based on requirements.
