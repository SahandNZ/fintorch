from fintorch.search.grid_search import GridSearch
from fintorch.search.parameter import SearchParameter


def objective_function(a: int, b: int) -> float:
    return a + b


def main():
    parameters = [SearchParameter(name="a", values=range(100)),
                  SearchParameter(name="b", values=range(200))]

    method = GridSearch(parameters=parameters, objective_function=objective_function)
    results = method.search()

    for result in results:
        print(result)


if __name__ == '__main__':
    main()
