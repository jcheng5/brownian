from shiny import reactive


def reactive_smooth(n_samples, smoother, *, filter_none=True):
    """Decorator for smoothing out reactive calculations over multiple samples"""

    def wrapper(calc):
        buffer = []  # Ring buffer of capacity `n_samples`
        result = reactive.Value(None)  # Holds the most recent smoothed result

        @reactive.Effect
        def _():
            # Get latest value. Because this is happening in a reactive Effect, we'll
            # automatically take a reactive dependency on whatever is happening in the
            # calc().
            new_value = calc()
            buffer.append(new_value)
            while len(buffer) > n_samples:
                buffer.pop(0)

            if not filter_none:
                result.set(smoother(buffer))
            else:
                # The filter cannot handle None values; remove any in the buffer
                filt_samples = [s for s in buffer if s is not None]
                if len(filt_samples) == 0:
                    result.set(None)
                else:
                    result.set(smoother(filt_samples))

        # The return value for the wrapper
        return result.get

    return wrapper
