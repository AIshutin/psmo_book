# Семинар

## Задачи

```{admonition} Задача 1
:class: attention

Правила работы с многомерными случайными величинами ($y$, $z$ -- случайные векторы, $A$, $B$ -- матрицы констант; считаем, что все размеры подходящие):
1. $\E(Ay) = A\E(y)$.
2. \Var(Ay) = A\Var(y)A'$.
3. $\Var(y + z) = \Var(y) + \Var(z) + \cov(y, z) + \cov(z, y)$.
4. $\cov(Ay, Bz) = A\cov(y, z)B'$.

Рассмотрим линейную модель $y = X\beta + u$, оцениваемую при помощи МНК. Пусть $\E(u) = 0$, $\Var(u) = \sigma^2 I$, число наблюдений равно $n$, число регрессоров, включая константный, равно $k$. 

Заполните матрицу характеристик элементов МНК:
$$
	\begin{center}
	\begin{tabular}{c|ccccc}
		$\Var(\cdot)$ & $y$ & $\hat{y}$ & $\hat{\beta}$ & $\hat{u}$ & $u$ \\
		\hline
		$y$ & $\ldots$ &&&& \\
		$\hat{y}$ &&$\ldots$&&& \\
		$\hat{\beta}$ &&&$\ldots$&& \\
		$\hat{u}$ &&&&$\ldots$& \\
		$u$ &&&&&$\ldots$ \\
		\hline
		$\E(\cdot)$ &&&&& \\
		\hline
	\end{tabular}
\end{center}
$$

	Для каждого элемента укажите размеры.
1. $\E(y)$
2. $\E(\hat{\beta})$
3. $\Var(y)$
4. $\Var(\hat{\beta})$
5. $\cov(\hat{\beta}, \hat{u})$
6. $\ldots$
```
