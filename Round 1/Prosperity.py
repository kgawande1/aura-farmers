def maximize_seashells():
    import math

    #Indexing currencies
    #0 = Snowball, 1 = Pizza, 2 = Silicon, 3 = SeaShell
    rates = [
        [1.00, 1.45, 0.52, 0.72],  #snowvall
        [0.70, 1.00, 0.31, 0.48],  #pizza
        [1.95, 3.10, 1.00, 1.49],  #silocone
        [1.34, 1.98, 0.64, 1.00]   #seashell
    ]
    
    currency_names = ["Snowball", "Pizza", "Silicon", "SeaShell"]

    start_currency = 3  
    end_currency   = 3  
    max_trades     = 5  

    best_product = -1.0
    best_path = None

    #depth first search
    def dfs(current_currency, step, current_product, path):
        nonlocal best_product, best_path

        if step == max_trades:
            if current_currency == end_currency:
                if current_product > best_product:
                    best_product = current_product
                    best_path = path[:]
            return

        for next_currency in range(4):
            new_product = current_product * rates[current_currency][next_currency]
            path.append(next_currency)
            dfs(next_currency, step + 1, new_product, path)
            path.pop()

    dfs(start_currency, 0, 1.0, [start_currency])


    print("Best product:", best_product)
    print("Best path (by indices):", best_path)
    if best_path:
        name_path = [currency_names[idx] for idx in best_path]
        print("Best path (by names):", " -> ".join(name_path))


if __name__ == "__main__":
    maximize_seashells()
